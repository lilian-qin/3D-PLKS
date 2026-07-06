from torch import nn
import torch
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch.nn import Module, Linear, MSELoss, ModuleList
from torch.nn.functional import relu
from torch.utils.data import WeightedRandomSampler
import pytorch_lightning as pl
from torch import optim
from torch_geometric.utils import dropout_adj
from torch_geometric.loader import DataLoader as GeoDataLoader  # Updated import
from pathlib import Path
import sys
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from base.dataset import ddgData, ddgDataSet,PLAffinityDataSet
from base.precomputed_dataset import PrecomputedDataset
from models.graphnorm.graphnorm import GraphNorm
from models.egnn.egnn import E_GCL, EGNN

from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter_mean
from torch_geometric.data import Data
import pandas as pd
import numpy as np

def to_np(x):
    return x.cpu().detach().numpy()


class ddgEGNN(pl.LightningModule):
    def __init__(
        self,
        num_node_features: int,
        loader_config: dict,
        dataset_config: dict,
        trainer_config: dict,
        num_edge_features: int = 0,
        embedding_in_nf: int = 32,
        embedding_out_nf: int = 32,
        egnn_layer_hidden_nfs: list = [32, 32, 32],
        num_classes: int = 1,
        opt: str = 'adam',
        loss: str = 'mse',
        scheduler: str = None,
        lr: float = 10e-3,
        dropout: float = 0.0,
        balanced_loss: bool = False,
        attention: bool = False,
        residual: bool = True,
        normalize: bool = False,
        tanh: bool = False,
        update_coords: bool = True,
        weight_decay: float = 0,
        norm: str = None,
        norm_nodes: str = None,
        pool_graphvectors: str = 'concat',
        **kwargs,
    ):
        super(ddgEGNN, self).__init__()

        self.loader_config = loader_config
        self.dataset_config = dataset_config
        self.trainer_config = trainer_config
        self.update_coords = update_coords
        self.embedding_out_nf = embedding_out_nf
        self.num_classes = num_classes
        self.dropout = dropout
        self.pool_graphvectors = pool_graphvectors

        self.embedding_in = Linear(num_node_features, embedding_in_nf)
        self.embedding_out = Linear(embedding_in_nf, embedding_out_nf)

        egnn_layers = []
        for hidden_nf in egnn_layer_hidden_nfs:
            layer = E_GCL(
                embedding_in_nf, hidden_nf, embedding_in_nf,
                edges_in_d=num_edge_features,
                act_fn=nn.SiLU(),
                attention=attention,
                residual=residual,
                normalize=normalize,
                coords_agg='mean',
                tanh=tanh,
                norm_nodes=norm_nodes,
            )
            egnn_layers.append(layer)
        self.egnn_layers = ModuleList(egnn_layers)

        # --- Prediction head input dim depends on pooling mode ---
        if pool_graphvectors == 'concat':
            head_input_dim = 2 * embedding_out_nf
        elif pool_graphvectors == 'hybrid':
            head_input_dim = 4 * embedding_out_nf  # [v0, v1, v0-v1, v0*v1]
        elif pool_graphvectors in ('diff', 'max', 'mean'):
            head_input_dim = embedding_out_nf
        else:
            raise ValueError(f"Unknown pool_graphvectors: {pool_graphvectors}")

        # --- Prediction head ---
        self.head_fc1 = nn.Linear(head_input_dim, 1024)
        self.head_fc2 = nn.Linear(1024, 512)
        self.head_out = nn.Linear(512, self.num_classes)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        # --- Training setup ---
        self.opt = opt
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        if loss == 'mse':
            self.loss_fn = MSELoss()
        else:
            raise NotImplementedError
        if balanced_loss:
            raise NotImplementedError

        self.test_set_predictions = []
        self.norm_nodes = norm_nodes
        if norm_nodes:
            self.graphnorm = GraphNorm(embedding_out_nf)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    def _pool_graph_vectors(self, graph_vectors):
        """Combine two graph embeddings based on pooling strategy."""
        v0, v1 = graph_vectors[0], graph_vectors[1]

        if self.pool_graphvectors == 'concat':
            return torch.cat([v0, v1], dim=1)
        elif self.pool_graphvectors == 'hybrid':
            return torch.cat([v0, v1, v0 - v1, v0 * v1], dim=1)
        elif self.pool_graphvectors == 'diff':
            return v0 - v1
        elif self.pool_graphvectors == 'max':
            return torch.amax(torch.stack([v0, v1], dim=0), dim=0)
        elif self.pool_graphvectors == 'mean':
            return torch.mean(torch.stack([v0, v1], dim=0), dim=0)


    def forward(self, graph):
        graph_vectors = []
        for g_ind in range(2):
            nodes = graph[f'x_{g_ind}'].float()
            edge_ind = graph[f'edge_index_{g_ind}']
            coords = graph[f'pos_{g_ind}'].float()
            edge_attr = graph[f'edge_attr_{g_ind}'].float()

            nodes = self.embedding_in(nodes)

            for egnn_layer in self.egnn_layers:
                edge_ind_post_dropout, edge_attr_post_dropout = dropout_adj(
                    edge_ind, edge_attr=edge_attr, p=self.dropout, training=self.training
                )
                if self.update_coords:
                    nodes, coords, _ = egnn_layer(
                        nodes, edge_ind_post_dropout, coords, edge_attr_post_dropout,
                        batch=graph[f'x_{g_ind}_batch']
                    )
                else:
                    nodes, _, _ = egnn_layer(
                        nodes, edge_ind_post_dropout, coords, edge_attr_post_dropout,
                        batch=graph[f'x_{g_ind}_batch']
                    )

            if self.norm_nodes:
                nodes = self.graphnorm(relu(self.embedding_out(nodes)), graph[f'x_{g_ind}_batch'])
            else:
                nodes = self.embedding_out(nodes)

            graph_vectors.append(global_max_pool(nodes, graph[f'x_{g_ind}_batch']))

        # --- Pool + Predict ---
        xc = self._pool_graph_vectors(graph_vectors)
        xc = self.dropout_layer(self.relu(self.head_fc1(xc)))
        xc = self.dropout_layer(self.relu(self.head_fc2(xc)))
        out = self.head_out(xc)
        return out


    def training_step(self, batch, batch_idx):
        """Run batch through forward and return loss, prediction and true label."""
        y = batch.y
        pred = self.forward(batch)
        if y.shape != pred.shape:
            try:
                y = y.view_as(pred)
            except Exception:
                print('Error in shape of labels vs pred')

        loss = self.loss_fn(pred, y.float())
        self.log('train_loss', loss)

        output = {'loss': loss, 'pred': pred, 'y': y}
        self.training_step_outputs.append(output)
        return output


    def validation_step(self, batch, batch_idx):
        y = batch.y
        pred = self.forward(batch)
        if y.shape != pred.shape:
            y = y.view_as(pred)

        loss = self.loss_fn(pred, y.float())
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        output = {'loss': loss, 'pred': pred, 'y': y}
        self.validation_step_outputs.append(output)
        return output


    def test_step(self, batch, batch_idx):
        y = batch.y
        pred = self.forward(batch)
        if y.shape != pred.shape:
            y = y.view_as(pred)

        loss = self.loss_fn(pred, y.float())

        # save predicted output
        output_preds = []
        labels = to_np(y).flatten()
        pred_np = to_np(pred.flatten())
        for ind, score in enumerate(pred_np):
            output_preds.append(
                (batch.pdb_wt[ind], batch.pdb_mut[ind], score, labels[ind])
            )
        self.test_set_predictions += output_preds
        self.log('test_loss', loss)

        output = {'loss': loss, 'pred': pred, 'y': y}
        self.test_step_outputs.append(output)
        return output


    def epoch_metrics(self, epoch_output):
        """Evaluate model performance with multiple metrics."""
        preds = []
        ys = []
        for step in epoch_output:
            pred = to_np(step['pred'].flatten())
            y = to_np(step['y'].flatten())
            preds += [i for i in pred]
            ys += [i for i in y]

        preds = np.array(preds)
        ys = np.array(ys)

        pearson_corr = pearsonr(ys, preds)[0]
        spearman_corr = spearmanr(ys, preds)[0]
        rmse = np.sqrt(mean_squared_error(ys, preds))
        mae = mean_absolute_error(ys, preds)

        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'rmse': rmse,
            'mae': mae,
        }


    def on_test_epoch_end(self):
        output = self.test_step_outputs
        metrics = self.epoch_metrics(output)
        print(f"[Test Pearson: {metrics['pearson']:.4f} | "
            f"Spearman: {metrics['spearman']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")
        self.log('test_pearson', metrics['pearson'])
        self.log('test_spearman', metrics['spearman'])
        self.log('test_rmse', metrics['rmse'])
        self.log('test_mae', metrics['mae'])
        self.test_step_outputs.clear()


    def on_validation_epoch_end(self):
        output = self.validation_step_outputs
        metrics = self.epoch_metrics(output)
        print(f"[Val Pearson: {metrics['pearson']:.4f} | "
            f"Spearman: {metrics['spearman']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

        # val_pearson uses the COMBINED prediction — ModelCheckpoint will track
        # the deployed model's actual metric, not just head_out alone.
        self.log('val_pearson', metrics['pearson'], prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_spearman', metrics['spearman'], prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_rmse', metrics['rmse'], prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_mae', metrics['mae'], prog_bar=True, on_epoch=True, on_step=False)
        self.validation_step_outputs.clear()


    def on_train_epoch_end(self):
        output = self.training_step_outputs
        metrics = self.epoch_metrics(output)
        print(f"[Train Pearson: {metrics['pearson']:.4f} | "
            f"Spearman: {metrics['spearman']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")
        self.log('train_pearson', metrics['pearson'])
        self.log('train_spearman', metrics['spearman'])
        self.log('train_rmse', metrics['rmse'])
        self.log('train_mae', metrics['mae'])
        self.training_step_outputs.clear()


    def save_test_predictions(self, filename: Path):
        """Save test predictions to csv file.

        pred_score is the COMBINED prediction (head_out + head_out2).
        """
        with open(filename, 'w') as outf:
            outf.write('wt_pdb,mut_pdb,pred_score,true_label\n')
            for p in self.test_set_predictions:
                outf.write(f'{p[0]},{p[1]},{p[2]},{p[3]}\n')

    def configure_optimizers(self):
        """Configure optimizers and define scheduler"""
        if self.opt == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        if not self.scheduler:
            return optimizer
        else:
            if self.scheduler == 'CosineAnnealingWarmRestarts':
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    self.trainer_config['max_epochs'],
                    eta_min=1e-4)
            elif self.scheduler == 'CosineAnnealing':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    self.trainer_config['max_epochs'],
                    eta_min=1e-6)
            elif self.scheduler == 'ReduceLROnPlateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss',  # required for ReduceLROnPlateau
                    },
                }
            else:
                raise NotImplementedError
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
            }

    def _make_loader(self, graph_dir: str, shuffle: bool):
        ds = PrecomputedDataset(graph_dir)
        batch_ls = ['x_0', 'x_1']
        loader = GeoDataLoader(
            ds,
            batch_size=self.loader_config['batch_size'],
            shuffle=shuffle,
            num_workers=self.loader_config['num_workers'],
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            follow_batch=batch_ls,
        )
        return loader

    def train_dataloader(self):
        if self.dataset_config['precomputed_dirs']['train'] is None:
            return None
        return self._make_loader(self.dataset_config['precomputed_dirs']['train'], shuffle=True)

    def val_dataloader(self):
        if self.dataset_config['precomputed_dirs']['val'] is None:
            return None
        return self._make_loader(self.dataset_config['precomputed_dirs']['val'], shuffle=False)

    def test_dataloader(self):
        if self.dataset_config['precomputed_dirs']['test'] is None:
            return None
        return self._make_loader(self.dataset_config['precomputed_dirs']['test'], shuffle=False)

    '''
    def test_dataloader(self):
        """Load and batch test data"""
        if self.dataset_config['input_files']['test'] is None:
            return None

        ds = ddgDataSet(
            interaction_dist=self.dataset_config['interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            cache_frames=self.dataset_config['cache_frames'],
        )
        for f in self.dataset_config['input_files']['test']:
            ds.populate(f, overwrite=False)

        batch_ls = ['x_0', 'x_1']
        loader = GeoDataLoader(
            ds,
            batch_size=self.loader_config['batch_size'],
            shuffle=False,
            num_workers=self.loader_config['num_workers'],
            follow_batch=batch_ls)
        return loader

    def train_dataloader(self):
        """Load and batch train data"""
        if self.dataset_config['input_files']['train'] is None:
            return None

        ds = ddgDataSet(
            interaction_dist=self.dataset_config['interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            cache_frames=self.dataset_config['cache_frames']
        )
        for f in self.dataset_config['input_files']['train']:
            ds.populate(f, overwrite=False)

        sampler = None

        batch_ls = ['x_0', 'x_1']
        loader = GeoDataLoader(
            ds,
            batch_size=self.loader_config['batch_size'],
            shuffle=True,  # Changed to True for training
            num_workers=self.loader_config['num_workers'],
            sampler=sampler,
            follow_batch=batch_ls)

        return loader

    def val_dataloader(self):
        """Load and batch val data"""
        if self.dataset_config['input_files']['val'] is None:
            return None

        ds = ddgDataSet(
            interaction_dist=self.dataset_config['interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            cache_frames=self.dataset_config['cache_frames'],
        )
        for f in self.dataset_config['input_files']['val']:
            ds.populate(f, overwrite=False)

        batch_ls = ['x_0', 'x_1']
        loader = GeoDataLoader(
            ds,
            batch_size=self.loader_config['batch_size'],
            shuffle=False,
            num_workers=self.loader_config['num_workers'],
            follow_batch=batch_ls)

        return loader
    '''
    


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    h = torch.ones(batch_size * n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)
    h, x = egnn(h, x, edges, edge_attr)
    print(h, x)