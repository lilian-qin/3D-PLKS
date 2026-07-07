# single_dataset.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from torch_geometric.utils import remove_self_loops

# Import your existing utils for type_map and one_hot
import utils
from torch_geometric.data import Batch


class GIGNData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('edge_index_intra', 'edge_index_inter'):
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class SingleComplexDataset(Dataset):
    """Single protein-ligand complex dataset for absolute pIC50 prediction."""

    def __init__(self, csv_path: str, interaction_dist: float = 5.0):
        self.df = pd.read_csv(csv_path)
        self.interaction_dist = interaction_dist
        self.type_map = utils.get_type_map()

    def __len__(self):
        return len(self.df)

    def _get_node_features(self, df):
        types = df['lmg_types'].apply(lambda x: self.type_map[x])
        types = np.array(types)
        types = utils.get_one_hot(types, nb_classes=max(self.type_map.values()) + 1)

        if 'formal_charge' in df.columns:
            formal_charge = df['formal_charge'].values.reshape(-1, 1).astype(np.float32)
        else:
            formal_charge = np.zeros((len(df), 1), dtype=np.float32)

        return np.concatenate([types, formal_charge], axis=-1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        parquet_path = row['complex_path']

        if not Path(parquet_path).exists():
            print(f"[WARN] Missing: {parquet_path}, skipping")
            return self.__getitem__((idx + 1) % len(self))

        complex_df = pd.read_parquet(parquet_path)
        prot_df = complex_df[complex_df['is_ligand'] == 0].reset_index(drop=True)
        lig_df = complex_df[complex_df['is_ligand'] == 1].reset_index(drop=True)

        coords_prot = prot_df[['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)
        coords_lig = lig_df[['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)

        if len(coords_prot) == 0 or len(coords_lig) == 0:
            return self.__getitem__((idx + 1) % len(self))

        # Find protein atoms near ligand
        dist_lig_prot = cdist(coords_lig, coords_prot)
        _, prot_near_idx = np.where(dist_lig_prot < self.interaction_dist)
        prot_node_idx = sorted(np.unique(prot_near_idx))

        if len(prot_node_idx) == 0:
            _, prot_near_idx = np.where(dist_lig_prot < 8.0)
            prot_node_idx = sorted(np.unique(prot_near_idx))
            if len(prot_node_idx) == 0:
                return self.__getitem__((idx + 1) % len(self))

        prot_nodes = prot_df.iloc[prot_node_idx]
        coords_prot = coords_prot[prot_node_idx]

        # Node features + protein/ligand flag
        node_feats_prot = self._get_node_features(prot_nodes)
        node_feats_lig = self._get_node_features(lig_df)

        flag_prot = np.zeros((len(coords_prot), 1))
        flag_lig = np.ones((len(coords_lig), 1))

        feats_prot = np.concatenate([node_feats_prot, flag_prot], axis=-1)
        feats_lig = np.concatenate([node_feats_lig, flag_lig], axis=-1)
        feats = np.concatenate([feats_prot, feats_lig], axis=0)
        coords = np.concatenate([coords_prot, coords_lig], axis=0)

        # Replace the edge construction section (from "# Edges" onward) with:

        # Edges
        lig_offset = len(prot_nodes)

        # Inter-molecular edges (protein <-> ligand)
        dist_inter = cdist(coords_prot, coords_lig)
        inter_src, inter_dst = np.where(dist_inter < self.interaction_dist)
        inter_src_full = np.concatenate([inter_src, inter_dst + lig_offset])
        inter_dst_full = np.concatenate([inter_dst + lig_offset, inter_src])
        edge_index_inter = torch.from_numpy(
            np.vstack([inter_src_full, inter_dst_full])
        ).long()

        # Intra-molecular edges (protein-protein + ligand-ligand)
        dist_intra_prot = cdist(coords_prot, coords_prot)
        intra_prot_src, intra_prot_dst = np.where(dist_intra_prot < self.interaction_dist)

        dist_intra_lig = cdist(coords_lig, coords_lig)
        intra_lig_src, intra_lig_dst = np.where(dist_intra_lig < self.interaction_dist)

        intra_src = np.concatenate([intra_prot_src, intra_lig_src + lig_offset])
        intra_dst = np.concatenate([intra_prot_dst, intra_lig_dst + lig_offset])
        edge_index_intra = torch.from_numpy(
            np.vstack([intra_src, intra_dst])
        ).long()

        # Remove self-loops
        edge_index_intra, _ = remove_self_loops(edge_index_intra)
        edge_index_inter, _ = remove_self_loops(edge_index_inter)

        graph = GIGNData(
            x=torch.from_numpy(feats).float(),
            edge_index_intra=edge_index_intra,
            edge_index_inter=edge_index_inter,
            pos=torch.from_numpy(coords).float(),
            y=torch.tensor([label]).float(),
            compound_id=row['compound_id'],
            kinase=row['kinase'],
        )
        return graph
        
    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)