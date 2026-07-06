from pathlib import Path
import struct
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch as th
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import time
import pandas as pd
import sys
from torch_geometric.utils import remove_self_loops
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data
from collections import defaultdict
import random


# import code needed for typing atoms
if __name__ == '__main__':
    import utils
    from atom_types.atom_types import Typer
else:
    try:
        from . import utils
        from atom_types.atom_types import Typer
    except:
        from base import utils
        from base.atom_types.atom_types import Typer


class ddgData(Data):
    """Aggregates two complex graphs (A & B), returning
        graph-specific edge indices, node features, edge attributes and node-coordinates,
        global ΔpIC50 label and paths to the complex parquet files.
    """

    def __init__(self, graph_list: list = None):
        super(ddgData, self).__init__()
        if graph_list is None:
            return

        for i in range(2):  # 0: complex A, 1: complex B
            self[f'edge_index_{i}'] = graph_list[i].edge_index
            self[f'x_{i}'] = graph_list[i].x
            self[f'edge_attr_{i}'] = graph_list[i].edge_attr
            self[f'pos_{i}'] = graph_list[i].pos

        self.y = graph_list[0].y  # ΔpIC50 label
        self.pdb_wt = graph_list[0].pdb_file   # complex A path (kept as pdb_wt for model compatibility)
        self.pdb_mut = graph_list[1].pdb_file   # complex B path (kept as pdb_mut for model compatibility)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_0':
            return self.x_0.size(0)
        if key == 'edge_index_1':
            return self.x_1.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ddgDataSet(Dataset):
    """Protein-Ligand selectivity dataset: predicts ΔpIC50 between two kinase complexes."""

    def __init__(self,
        interaction_dist: float = 5,
        typing_mode: str = 'lmg',
        cache_frames: bool = False,
        **kwargs):

        self.type_map = utils.get_type_map()
        if typing_mode == 'lmg':
            self.node_feature_size = 13  # 11 atom types + 1 formal_charge + 1 protein/ligand flag
        self.entries = []
        self.labels = []
        self.edge_dim = 2
        self.interaction_dist = interaction_dist
        self.typing_mode = typing_mode
        self.cache = {}
        self.cache_frames = cache_frames

    def populate(self, input_file: Path, overwrite: bool = False):
        """Load CSV with columns: compound_id, kinase_A, kinase_B, complexA_path, complexB_path, label"""
        inf = pd.read_csv(input_file)
        labels = inf["label"].tolist()

        entries = []
        for _, row in inf.iterrows():
            entry = {
                'compound_id': row['compound_id'],
                'kinase_A': row['kinase_A'],
                'kinase_B': row['kinase_B'],
                'complexA_path': row['complexA_path'],
                'complexB_path': row['complexB_path'],
            }
            entries.append(entry)

        if overwrite:
            self.entries = entries
            self.labels = labels
        else:
            self.entries += entries
            self.labels += labels

    def __len__(self):
        return len(self.labels)

    def _get_node_features(self, df: pd.DataFrame):
        if self.typing_mode == 'lmg':
            # One-hot atom types (11 features)
            types = df['lmg_types'].apply(lambda x: self.type_map[x])
            types = np.array(types)
            types = utils.get_one_hot(types, nb_classes=max(self.type_map.values()) + 1)
            
            # Formal charge (1 feature)
            if 'formal_charge' in df.columns:
                formal_charge = df['formal_charge'].values.reshape(-1, 1).astype(np.float32)
            else:
                formal_charge = np.zeros((len(df), 1), dtype=np.float32)
            
            # Concatenate: [11 atom types | 1 formal_charge] = 12 features
            node_feats = np.concatenate([types, formal_charge], axis=-1)
            return node_feats
        else:
            raise NotImplementedError(self.typing_mode)
        
    def __aggregate_graphs__(self, graph_list: list):
        """Aggregate two graphs using ddgData class (same as original)."""
        aggregated = ddgData(graph_list)
        return aggregated

    def _build_graph(self, prot_df: pd.DataFrame, lig_df: pd.DataFrame):
        """Build protein-ligand interaction graph.
        Ligand is the center; protein atoms within interaction_dist are included."""

        coords_prot = prot_df[['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)
        coords_lig = lig_df[['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)

        if len(coords_prot) == 0 or len(coords_lig) == 0:
            return None

        # Find protein atoms near ligand
        dist_lig_prot = cdist(coords_lig, coords_prot)
        _, prot_near_idx = np.where(dist_lig_prot < self.interaction_dist)
        prot_node_idx = sorted(np.unique(prot_near_idx))

        if len(prot_node_idx) == 0:
            _, prot_near_idx = np.where(dist_lig_prot < 8.0)
            prot_node_idx = sorted(np.unique(prot_near_idx))
            if len(prot_node_idx) == 0:
                return None

        prot_nodes = prot_df.iloc[prot_node_idx]
        coords_prot = coords_prot[prot_node_idx]

        # Node features
        node_feats_prot = self._get_node_features(prot_nodes)
        node_feats_lig = self._get_node_features(lig_df)

        flag_prot = np.zeros((len(coords_prot), 1))  # 0 = protein
        flag_lig = np.ones((len(coords_lig), 1))      # 1 = ligand

        feats_prot = np.concatenate([coords_prot, node_feats_prot, flag_prot], axis=-1)
        feats_lig = np.concatenate([coords_lig, node_feats_lig, flag_lig], axis=-1)
        feats = np.concatenate([feats_prot, feats_lig], axis=0)

        # Edges
        lig_offset = len(prot_nodes)

        # Inter: protein <-> ligand
        dist_inter = cdist(coords_prot, coords_lig)
        inter_src, inter_dst = np.where(dist_inter < self.interaction_dist)

        # Intra-protein
        dist_intra_prot = cdist(coords_prot, coords_prot)
        intra_prot_src, intra_prot_dst = np.where(dist_intra_prot < self.interaction_dist)

        # Intra-ligand
        dist_intra_lig = cdist(coords_lig, coords_lig)
        intra_lig_src, intra_lig_dst = np.where(dist_intra_lig < self.interaction_dist)

        edge_src = np.concatenate([inter_src, intra_prot_src, intra_lig_src + lig_offset])
        edge_dst = np.concatenate([inter_dst + lig_offset, intra_prot_dst, intra_lig_dst + lig_offset])

        edge_src_full = np.concatenate([edge_src, edge_dst])
        edge_dst_full = np.concatenate([edge_dst, edge_src])
        edge_indices = np.vstack([edge_src_full, edge_dst_full]).astype(np.float64)

        edge_attr = np.concatenate([
            np.ones(len(inter_src)),    # inter = 1
            np.zeros(len(intra_prot_src)),  # intra = 0
            np.zeros(len(intra_lig_src)),
        ])
        edge_attr_full = np.expand_dims(np.concatenate([edge_attr, edge_attr]), 1)

        return feats, edge_indices, edge_attr_full

    def _load_and_build_graph(self, parquet_path: str, label, entry, graph_id: str):
        """Load parquet, split protein/ligand, build graph, return Data object."""

        # Cache check
        if self.cache_frames and parquet_path in self.cache:
            complex_df = self.cache[parquet_path].copy()
        elif Path(parquet_path).exists():
            complex_df = pd.read_parquet(parquet_path)
        else:
            print(f"[WARN] Parquet not found: {parquet_path}")
            return None

        if self.cache_frames and parquet_path not in self.cache:
            self.cache[parquet_path] = complex_df.copy()

        # Split
        prot_df = complex_df[complex_df['is_ligand'] == 0].reset_index(drop=True)
        lig_df = complex_df[complex_df['is_ligand'] == 1].reset_index(drop=True)

        result = self._build_graph(prot_df, lig_df)
        if result is None:
            return None

        feats, edge_indices, edge_attr = result

        edge_index, edge_attr_tensor = remove_self_loops(
            edge_index=th.from_numpy(edge_indices).long(),
            edge_attr=th.from_numpy(edge_attr)
        )

        graph = Data(
            x=th.from_numpy(feats[:, 3:]).float(),
            edge_index=edge_index,
            edge_attr=edge_attr_tensor.float(),
            pos=th.from_numpy(feats[:, :3]).float(),
            y=th.tensor(label).float(),
            pdb_file=parquet_path,  # keep for compatibility with ddgData
            wt_mut=graph_id,        # 'A' or 'B' instead of 'wt'/'mut'
        )
        return graph

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        entry = self.entries[idx]

        # Build graph for complex A (analogous to "wt")
        graph_A = self._load_and_build_graph(
            entry['complexA_path'], label, entry, graph_id='A'
        )
        if graph_A is None:
            print(f"[WARN] Skipping idx={idx}, complex A failed")
            return self.__getitem__((idx + 1) % len(self))

        # Build graph for complex B (analogous to "mut")
        graph_B = self._load_and_build_graph(
            entry['complexB_path'], label, entry, graph_id='B'
        )
        if graph_B is None:
            print(f"[WARN] Skipping idx={idx}, complex B failed")
            return self.__getitem__((idx + 1) % len(self))

        # Aggregate — same as original wt+mut aggregation
        graph = self.__aggregate_graphs__([graph_A, graph_B])
        return graph

class PLAffinityDataSet(Dataset):
    """Single protein-ligand complex → pIC50 prediction."""

    def __init__(self, interaction_dist=5, typing_mode='lmg', cache_frames=False, **kwargs):
        self.type_map = utils.get_type_map()
        self.node_feature_size = 12
        self.entries = []
        self.labels = []
        self.edge_dim = 2
        self.interaction_dist = interaction_dist
        self.typing_mode = typing_mode
        self.cache = {}
        self.cache_frames = cache_frames

    def populate(self, input_file: Path, overwrite=False):
        """CSV columns: parquet_path, pIC50"""
        inf = pd.read_csv(input_file)
        labels = inf["label"].tolist()
        entries = [{'parquet_path': row['complex_path']} for _, row in inf.iterrows()]

        if overwrite:
            self.entries, self.labels = entries, labels
        else:
            self.entries += entries
            self.labels += labels

    def __len__(self):
        return len(self.labels)

    def _get_node_features(self, df):
        types = df['lmg_types'].apply(lambda x: self.type_map[x])
        return utils.get_one_hot(np.array(types), nb_classes=max(self.type_map.values()) + 1)

    def _build_graph(self, prot_df, lig_df):
        coords_prot = prot_df[['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)
        coords_lig = lig_df[['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)

        if len(coords_prot) == 0 or len(coords_lig) == 0:
            return None

        dist_lig_prot = cdist(coords_lig, coords_prot)
        _, prot_near_idx = np.where(dist_lig_prot < self.interaction_dist)
        prot_node_idx = sorted(np.unique(prot_near_idx))

        if len(prot_node_idx) == 0:
            return None

        prot_nodes = prot_df.iloc[prot_node_idx]
        coords_prot = coords_prot[prot_node_idx]

        node_feats_prot = self._get_node_features(prot_nodes)
        node_feats_lig = self._get_node_features(lig_df)

        flag_prot = np.zeros((len(coords_prot), 1))
        flag_lig = np.ones((len(coords_lig), 1))

        feats = np.concatenate([
            np.concatenate([coords_prot, node_feats_prot, flag_prot], axis=-1),
            np.concatenate([coords_lig, node_feats_lig, flag_lig], axis=-1),
        ], axis=0)

        lig_offset = len(prot_nodes)

        dist_inter = cdist(coords_prot, coords_lig)
        inter_src, inter_dst = np.where(dist_inter < self.interaction_dist)

        dist_intra_prot = cdist(coords_prot, coords_prot)
        intra_prot_src, intra_prot_dst = np.where(dist_intra_prot < self.interaction_dist)

        dist_intra_lig = cdist(coords_lig, coords_lig)
        intra_lig_src, intra_lig_dst = np.where(dist_intra_lig < self.interaction_dist)

        edge_src = np.concatenate([inter_src, intra_prot_src, intra_lig_src + lig_offset])
        edge_dst = np.concatenate([inter_dst + lig_offset, intra_prot_dst, intra_lig_dst + lig_offset])

        edge_src_full = np.concatenate([edge_src, edge_dst])
        edge_dst_full = np.concatenate([edge_dst, edge_src])
        edge_indices = np.vstack([edge_src_full, edge_dst_full]).astype(np.float64)

        edge_attr = np.concatenate([
            np.ones(len(inter_src)),
            np.zeros(len(intra_prot_src)),
            np.zeros(len(intra_lig_src)),
        ])
        edge_attr_full = np.expand_dims(np.concatenate([edge_attr, edge_attr]), 1)

        return feats, edge_indices, edge_attr_full

    def __getitem__(self, idx):
        label = self.labels[idx]
        entry = self.entries[idx]
        parquet_path = entry['parquet_path']

        if self.cache_frames and parquet_path in self.cache:
            complex_df = self.cache[parquet_path].copy()
        elif Path(parquet_path).exists():
            complex_df = pd.read_parquet(parquet_path)
        else:
            return self.__getitem__((idx + 1) % len(self))

        if self.cache_frames and parquet_path not in self.cache:
            self.cache[parquet_path] = complex_df.copy()

        prot_df = complex_df[complex_df['is_ligand'] == 0].reset_index(drop=True)
        lig_df = complex_df[complex_df['is_ligand'] == 1].reset_index(drop=True)

        result = self._build_graph(prot_df, lig_df)
        if result is None:
            return self.__getitem__((idx + 1) % len(self))

        feats, edge_indices, edge_attr = result

        edge_index, edge_attr_tensor = remove_self_loops(
            edge_index=th.from_numpy(edge_indices).long(),
            edge_attr=th.from_numpy(edge_attr)
        )

        return Data(
            x=th.from_numpy(feats[:, 3:]).float(),
            edge_index=edge_index,
            edge_attr=edge_attr_tensor.float(),
            pos=th.from_numpy(feats[:, :3]).float(),
            y=th.tensor(label).float(),
            pdb_file=parquet_path,
        )    
