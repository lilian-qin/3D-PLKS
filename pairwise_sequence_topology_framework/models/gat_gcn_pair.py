import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, max_seq_len=108,dropout=0.2,
                 pool_method='diff'):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.pool_method = pool_method
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ===== Ligand GAT + GCN layers =====
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)

        # ===== Protein sequence encoder (weight-shared) =====
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.max_seq_len = max_seq_len
        self.conv_xt_1 = nn.Conv1d(in_channels=max_seq_len, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(n_filters * (embed_dim - 7), output_dim)  # 32 * 121 = 3872
        #self.fc1_xt = nn.Linear(32 * (max_seq_len - 7), output_dim)

        # ===== Comparison + MLP head =====
        pair_dim = 2 * output_dim  # 256

        if pool_method == 'diff':
            mlp_input_dim = pair_dim
        elif pool_method == 'diff_lig':
            mlp_input_dim = pair_dim
        elif pool_method == 'concat_just':
            mlp_input_dim = 2 * pair_dim
        elif pool_method == 'concat_full':
            mlp_input_dim = 4 * pair_dim
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")

        self.fc1 = nn.Linear(mlp_input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def encode_ligand(self, x, edge_index, batch):
        """Encode ligand graph through GAT + GCN layers. Called once."""
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        return x

    def encode_protein(self, target):
        target = target.view(-1, self.max_seq_len).long()  # int16 -> long for embedding
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)
        return xt

    def forward(self, data):
        """
        Siamese forward pass.
        Same ligand graph, two different protein sequences.
        Predicts ΔpIC50 = pIC50_A - pIC50_B.
        """
        # ===== Encode ligand once =====
        x_lig = self.encode_ligand(data.x, data.edge_index, data.batch)

        # ===== Encode two proteins =====
        xt_A = self.encode_protein(data.target_A)
        xt_B = self.encode_protein(data.target_B)

        # ===== Compare =====
        if self.pool_method == 'diff':
            h_A = torch.cat((x_lig, xt_A), dim=1)
            h_B = torch.cat((x_lig, xt_B), dim=1)
            h = h_A - h_B
        elif self.pool_method == 'diff_lig':
            h = torch.cat((x_lig, xt_A - xt_B), dim=1)
        elif self.pool_method == 'concat_just':
            h_A = torch.cat((x_lig, xt_A), dim=1)
            h_B = torch.cat((x_lig, xt_B), dim=1)
            h = torch.cat((h_A, h_B), dim=1)
        elif self.pool_method == 'concat_full':
            h_A = torch.cat((x_lig, xt_A), dim=1)
            h_B = torch.cat((x_lig, xt_B), dim=1)
            h = torch.cat((h_A, h_B, h_A - h_B, h_A * h_B), dim=1)

        # ===== MLP head =====
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))
        h = self.dropout(h)
        out = self.out(h)
        return out