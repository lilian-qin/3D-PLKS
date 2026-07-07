import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
from dgllife.model.gnn import GAT, AttentiveFPGNN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.predict = nn.ModuleList()
        for j in range(n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(d_graph_layer, d_FC_layer))
                self.predict.append(nn.Dropout(dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == n_FC_layer - 1:
                self.predict.append(nn.Linear(d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(d_FC_layer, d_FC_layer))
                self.predict.append(nn.Dropout(dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h


class DTIConvGraph3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3, self).__init__()
        self.mpl = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim), nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim), nn.LeakyReLU(),
        )

    def forward(self, bg, atom_feats, bond_feats):
        with bg.local_scope():
            bg.ndata['h'] = atom_feats
            bg.edata['e'] = bond_feats
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            e = self.mpl(torch.cat([bg.edata['e'], bg.edata['m']], dim=1))
            return e


class DTIConvGraph3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(DTIConvGraph3Layer, self).__init__()
        self.grah_conv = DTIConvGraph3(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        return self.bn_layer(self.dropout(self.grah_conv(bg, atom_feats, bond_feats)))


class EdgeWeightAndSum(nn.Module):
    def __init__(self, in_feats):
        super(EdgeWeightAndSum, self).__init__()
        self.atom_weighting = nn.Sequential(nn.Linear(in_feats, 1), nn.Tanh())

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['w'] = self.atom_weighting(edge_feats)
            return dgl.sum_edges(g, 'e', 'w')


class EdgeWeightedSumAndMax(nn.Module):
    def __init__(self, in_feats):
        super(EdgeWeightedSumAndMax, self).__init__()
        self.weight_and_sum = EdgeWeightAndSum(in_feats)

    def forward(self, bg, edge_feats):
        h_g_sum = self.weight_and_sum(bg, edge_feats)
        with bg.local_scope():
            bg.edata['e'] = edge_feats
            h_g_max = dgl.max_edges(bg, 'e')
        return torch.cat([h_g_sum, h_g_max], dim=1)


class AttentiveGRU1(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()
        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size),
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """
        Does not touch g.ndata/g.edata — takes all needed tensors as arguments
        and uses update_all with explicit src/dst to avoid any scope issues.
        """
        with g.local_scope():
            # softmax-weighted, transformed edge messages
            g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
            g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'c'))
            context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()
        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size),
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        with g.local_scope():
            g.edata['a']  = edge_softmax(g, edge_logits)
            g.ndata['hv'] = self.project_node(node_feats)
            g.update_all(fn.u_mul_e('hv', 'a', 'm'), fn.sum('m', 'c'))
            context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()
        self.project_node  = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size), nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size), nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU(),
        )
        self.attentive_gru = AttentiveGRU1(
            graph_feat_size, graph_feat_size, graph_feat_size, dropout
        )

    def forward(self, g, node_feats, edge_feats):
        hv     = node_feats
        hv_new = self.project_node(node_feats)

        with g.local_scope():
            g.ndata['hv']     = hv
            g.ndata['hv_new'] = hv_new
            g.edata['he']     = edge_feats

            g.apply_edges(lambda e: {'he1': torch.cat([e.src['hv'], e.data['he']], dim=1)})
            he1 = self.project_edge1(g.edata['he1'])
            g.edata['he1'] = he1

            g.apply_edges(lambda e: {'he2': torch.cat([e.dst['hv_new'], e.data['he1']], dim=1)})
            
            # .clone() forces a real memory copy before local_scope releases its storage
            he2 = g.edata['he2'].clone()
            he1 = g.edata['he1'].clone()

        #print("he2 shape:", he2.shape)
        #print(self.project_edge2)
        logits = self.project_edge2(he2)
        return self.attentive_gru(g, logits, he1, hv_new)


class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()
        self.project_edge  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU(),
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer      = nn.BatchNorm1d(graph_feat_size)

    def forward(self, g, node_feats):
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g.apply_edges(lambda e: {'he': torch.cat([e.dst['hv'], e.src['hv']], dim=1)})
            he = g.edata['he'].clone()   # clone before scope exits

        logits = self.project_edge(he)
        return self.bn_layer(self.attentive_gru(g, logits, node_feats))


class ModifiedAttentiveFPGNNV2(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers=2,
                 graph_feat_size=200, dropout=0.):
        super(ModifiedAttentiveFPGNNV2, self).__init__()
        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers   = nn.ModuleList([
            GNNLayer(graph_feat_size, graph_feat_size, dropout)
            for _ in range(num_layers - 1)
        ])

    def forward(self, g, node_feats, edge_feats):
        node_feats      = self.init_context(g, node_feats, edge_feats)
        sum_node_feats  = node_feats
        for gnn in self.gnn_layers:
            node_feats     = gnn(g, node_feats)
            sum_node_feats = sum_node_feats + node_feats
        return sum_node_feats


class ModifiedAttentiveFPPredictorV2(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers=2,
                 graph_feat_size=200, dropout=0.):
        super(ModifiedAttentiveFPPredictorV2, self).__init__()
        self.gnn = ModifiedAttentiveFPGNNV2(
            node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
            num_layers=num_layers, graph_feat_size=graph_feat_size, dropout=dropout,
        )

    def forward(self, g, node_feats, edge_feats):
        return self.gnn(g, node_feats, edge_feats)


class DTIPredictorV4_V2(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size,
                 outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(DTIPredictorV4_V2, self).__init__()

        # intra-molecular graph (g): node_feat_size nodes, edge_feat_size edges
        self.cov_graph = ModifiedAttentiveFPPredictorV2(
            node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout
        )
        # inter-molecular graph (g3): graph_feat_size node feats + 1-dim dist edge
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size + 1, outdim_g3, dropout)

        self.readout = EdgeWeightedSumAndMax(outdim_g3)
        self.FC      = FC(outdim_g3 * 2, d_FC_layer, n_FC_layer, dropout, n_tasks)

    def forward(self, bg, bg3):
        atom_feats  = bg.ndata.pop('h')        # [N, node_feat_size]
        bond_feats  = bg.edata.pop('e')        # [E, edge_feat_size]
        atom_feats  = self.cov_graph(bg, atom_feats, bond_feats)   # [N, graph_feat_size]
        bond_feats3 = bg3.edata['e']           # [E3, 1]
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)  # [E3, outdim_g3]
        readouts    = self.readout(bg3, bond_feats3)                    # [batch, outdim_g3*2]
        return self.FC(readouts)                                        # [batch, n_tasks]