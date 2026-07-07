import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, max_seq_len=108, dropout=0.2,
                 pool_method='diff'):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        self.pool_method = pool_method

        # ===== Ligand GIN layers =====
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # ===== Protein sequence encoder (weight-shared) =====
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.max_seq_len = max_seq_len
        self.conv_xt_1 = nn.Conv1d(in_channels=max_seq_len, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(n_filters * (embed_dim - 7), output_dim)  # 32 * 121 = 3872
        #self.fc1_xt = nn.Linear(32 * (max_seq_len - 7), output_dim)

        # ===== Comparison + MLP head =====
        # h_A and h_B are each [B, output_dim*2] = [B, 256]
        if pool_method == 'diff':
            mlp_input_dim = 2 * output_dim  # 256, but ligand part cancels out

        elif pool_method == 'concat_just':
            mlp_input_dim = 4 * output_dim  # 512, concat h_A and h_B
        elif pool_method == 'concat_full':
            mlp_input_dim = 8 * output_dim  # 1024, [h_A, h_B, h_A-h_B, h_A*h_B]
        elif pool_method == 'diff_lig':
            mlp_input_dim = 2 * output_dim  # 256, diff ligand part
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")

        self.fc1 = nn.Linear(mlp_input_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def encode_ligand(self, x, edge_index, batch):
        """Encode ligand graph through GIN layers. Called once."""
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return x

    def encode_protein(self, target):
        target = target.view(-1, self.max_seq_len).long()  # int16 -> long for embedding
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        #xt = conv_xt.view(-1, 32 * (self.max_seq_len - 7))
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

        # ===== Combine ligand + protein for each arm =====
        h_A = torch.cat((x_lig, xt_A), dim=1)  # [B, 256]
        h_B = torch.cat((x_lig, xt_B), dim=1)  # [B, 256]

        # ===== Compare =====
        if self.pool_method == 'diff':
            h = h_A - h_B
        elif self.pool_method == 'diff_lig':
            h = torch.cat((x_lig, xt_A - xt_B), dim=1)
        elif self.pool_method == 'concat_just':
            h = torch.cat((h_A, h_B), dim=1)
        elif self.pool_method == 'concat_full':
            h = torch.cat((h_A, h_B, h_A - h_B, h_A * h_B), dim=1)

        # ===== MLP head =====
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)

        out = self.out(h)
        return out