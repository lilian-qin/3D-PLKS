"""
ign_dataset.py
--------------
IGN-compatible dataset driven entirely by the pre-built parquet files.

Graph schema
------------
g   – intra-molecular DGL graph  (protein + ligand atoms)
      ndata['h']  : node features  (type one-hot + formal_charge)  [N, node_feat_size]
      edata['e']  : [is_inter=0(1) | dist×0.1(1) | 3D-geom(9)] = 11-dim

g3  – inter-molecular DGL graph  (ligand ↔ pocket, dist < 5.0 Å)
      edata['e']  : dist × 0.1  [E, 1]

Distance thresholds
-------------------
_COVALENT_DIST = 2.0 Å  – intra edges (approximates covalent bonds)
inter_dist     = 5.0 Å  – inter edges (matches other models in pipeline)

Cache layout
------------
cache_dir/
    graph_dic/  {pair_id}.pkl   – per-complex DGL graphs (written once, never deleted)
    keys.bin                    – list of pair_ids (tiny, used as index)
    labels.bin                  – list of pIC50 labels

On second run: only keys.bin + labels.bin are loaded at startup.
Graphs are read from graph_dic/{key}.pkl on demand in __getitem__.
No RAM spike from loading all graphs at once.
"""

import os
import pickle
import warnings
import multiprocessing
from functools import partial
from itertools import repeat

import numpy as np
import pandas as pd
import torch
import dgl
from scipy.spatial.distance import cdist

import utils  # get_type_map(), get_one_hot()

warnings.filterwarnings('ignore')

# Covalent-bond distance upper bound — not a tunable hyperparameter
_COVALENT_DIST = 2.0


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def _get_node_features(df, type_map):
    """[N, n_types + 1] float32 — type one-hot | formal_charge"""
    types   = df['lmg_types'].apply(lambda x: type_map[x]).to_numpy()
    one_hot = utils.get_one_hot(types, nb_classes=max(type_map.values()) + 1)
    fc = (df['formal_charge'].values.reshape(-1, 1).astype(np.float32)
          if 'formal_charge' in df.columns
          else np.zeros((len(df), 1), dtype=np.float32))
    return np.concatenate([one_hot, fc], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# 3-D geometric edge features  (identical to original IGN)
# ---------------------------------------------------------------------------

def D3_info(a, b, c):
    ab = b - a
    ac = c - a
    cos_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac) + 1e-8)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    area  = 0.5 * np.linalg.norm(ab) * np.linalg.norm(ac) * np.sin(angle)
    return np.degrees(angle), area, np.linalg.norm(ac)


def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles, Areas, Distances = [], [], []
        for node_id in nodes_ls[2:]:
            angle, area, dist = D3_info(
                g.ndata['pos'][nodes_ls[0]].numpy(),
                g.ndata['pos'][nodes_ls[1]].numpy(),
                g.ndata['pos'][node_id].numpy(),
            )
            Angles.append(angle); Areas.append(area); Distances.append(dist)
        return [
            np.max(Angles)    * 0.01, np.sum(Angles)    * 0.01, np.mean(Angles)    * 0.01,
            np.max(Areas),            np.sum(Areas),             np.mean(Areas),
            np.max(Distances) * 0.1,  np.sum(Distances) * 0.1,  np.mean(Distances) * 0.1,
        ]
    return [0, 0, 0, 0, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Per-complex graph builder  (worker function)
# ---------------------------------------------------------------------------

def build_graph_from_parquet(
    parquet_path: str,
    key: str,
    label: float,
    graph_dic_path: str,
    type_map: dict,
    inter_dist: float,
    pocket_residue_dist: float = 5.0,
):
    """
    Build IGN g + g3 from a parquet file and write {key}.pkl.

    Protein is trimmed to the pocket: residues with any atom within
    `pocket_residue_dist` Å of any ligand atom are kept intact; the rest
    are dropped.
    """
    out_path = os.path.join(graph_dic_path, key + '.pkl')
    if os.path.exists(out_path):
        return

    try:
        df = pd.read_parquet(parquet_path)

        prot_df = df[df['is_ligand'] == 0].reset_index(drop=True)
        lig_df  = df[df['is_ligand'] == 1].reset_index(drop=True)

        if len(prot_df) == 0 or len(lig_df) == 0:
            raise ValueError('Empty protein or ligand block')

        # ── Build composite residue key: chain_id + residue_number + insertion ──
        # insertion is usually '' but can be 'A', 'B', etc. for inserted residues.
        # Fill NaN with '' so the concat is well-defined.
        ins = prot_df['insertion'].fillna('').astype(str)
        res_key = (
            prot_df['chain_id'].astype(str)
            + '_'
            + prot_df['residue_number'].astype(str)
            + '_'
            + ins
        ).values

        # ── Find pocket residues: any atom within cutoff of any ligand atom ──
        coords_prot_full = prot_df[['x_coord', 'y_coord', 'z_coord']].to_numpy(np.float32)
        coords_lig       = lig_df [['x_coord', 'y_coord', 'z_coord']].to_numpy(np.float32)

        d_prot_lig = cdist(coords_prot_full, coords_lig).min(axis=1)
        close_atom_mask = d_prot_lig < pocket_residue_dist
        pocket_residues = set(res_key[close_atom_mask])

        if not pocket_residues:
            raise ValueError(
                f'No protein residues within {pocket_residue_dist} Å of ligand'
            )

        # keep all atoms belonging to those residues (intact residues only)
        keep_mask = np.isin(res_key, list(pocket_residues))
        prot_df = prot_df[keep_mask].reset_index(drop=True)

        # ── Everything below is unchanged from the original code ──
        coords_prot = prot_df[['x_coord', 'y_coord', 'z_coord']].to_numpy(np.float32)
        num_prot    = len(coords_prot)
        num_lig     = len(coords_lig)
        num_atoms   = num_prot + num_lig

        # node features
        node_feats = np.concatenate([
            _get_node_features(prot_df, type_map),
            _get_node_features(lig_df,  type_map),
        ], axis=0)

        # intra edges: ~covalent bonds via 2.0 Å
        dp = cdist(coords_prot, coords_prot)
        pp_src, pp_dst = np.where((dp > 0) & (dp < _COVALENT_DIST))
        pp_d = dp[pp_src, pp_dst]

        dl = cdist(coords_lig, coords_lig)
        ll_src_r, ll_dst_r = np.where((dl > 0) & (dl < _COVALENT_DIST))
        ll_d   = dl[ll_src_r, ll_dst_r]
        ll_src = ll_src_r + num_prot
        ll_dst = ll_dst_r + num_prot

        intra_src = np.concatenate([pp_src, ll_src])
        intra_dst = np.concatenate([pp_dst, ll_dst])
        intra_d   = np.concatenate([pp_d,   ll_d])

        # inter edges: 5.0 Å, bidirectional
        di = cdist(coords_prot, coords_lig)
        p_idx, l_idx = np.where(di < inter_dist)

        if len(p_idx) == 0:
            raise ValueError(f'No inter-molecular contacts within {inter_dist} Å')

        inter_src_g3 = np.concatenate([p_idx,            l_idx + num_prot])
        inter_dst_g3 = np.concatenate([l_idx + num_prot, p_idx])
        inter_d_g3   = np.concatenate([di[p_idx, l_idx], di[p_idx, l_idx]])

        # build g (intra)
        g = dgl.graph((intra_src, intra_dst), num_nodes=num_atoms)
        g.ndata['h']   = torch.tensor(node_feats, dtype=torch.float)
        g.ndata['pos'] = torch.tensor(
            np.concatenate([coords_prot, coords_lig], axis=0), dtype=torch.float
        )
        g.edata['e'] = torch.cat([
            torch.zeros(len(intra_src), 1),
            torch.tensor(intra_d, dtype=torch.float).view(-1, 1) * 0.1,
        ], dim=-1)

        # 3-D geometric features
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()

        neighbors_ls = []
        for i, src in enumerate(src_nodes):
            tmp  = [src, dst_nodes[i]]
            nbrs = g.predecessors(src).tolist()
            if dst_nodes[i] in nbrs:
                nbrs.remove(dst_nodes[i])
            tmp.extend(nbrs)
            neighbors_ls.append(tmp)

        D3_th = torch.tensor(
            list(map(partial(D3_info_cal, g=g), neighbors_ls)), dtype=torch.float
        )

        if torch.any(torch.isnan(D3_th)):
            raise ValueError('NaN in 3D geometric features')

        g.edata['e'] = torch.cat([g.edata['e'], D3_th], dim=-1)
        g.ndata.pop('pos')

        # build g3 (inter)
        g3 = dgl.graph((inter_src_g3, inter_dst_g3), num_nodes=num_atoms)
        g3.edata['e'] = torch.tensor(inter_d_g3, dtype=torch.float).view(-1, 1) * 0.1

    except Exception as exc:
        print(f'[ERROR] {key}: {exc}')
        return

    with open(out_path, 'wb') as f:
        pickle.dump({'g': g, 'g3': g3, 'key': key, 'label': label}, f)


# ---------------------------------------------------------------------------
# collate_fn  (identical to original IGN)
# ---------------------------------------------------------------------------

def collate_fn_v2_MulPro(data_batch):
    graphs, graphs3, Ys, keys = map(list, zip(*data_batch))
    bg  = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys  = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg, bg3, Ys, keys


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IGNDataset:
    """
    IGN dataset driven by split CSVs from the kinase parquet pipeline.

    Startup behaviour
    -----------------
    First run  : builds per-complex .pkl files, saves keys.bin + labels.bin
    Later runs : loads keys.bin + labels.bin only (instant),
                 graphs read on demand from .pkl in __getitem__

    Parameters
    ----------
    csv_path    : split CSV with columns: compound_id, kinase, label, complex_path
    cache_dir   : root for this split's cache (use separate dirs per split)
    inter_dist  : Å cutoff for inter-molecular edges (default 5.0)
    num_process : parallel workers for graph generation
    rebuild     : if True, delete existing .pkl files and rebuild from parquets
    """

    def __init__(
        self,
        csv_path: str,
        cache_dir: str,
        inter_dist: float = 5.0,
        pocket_residue_dist: float = 5.0,
        num_process: int = 8,
        rebuild: bool = False,
    ):
        self.cache_dir      = cache_dir
        self.inter_dist     = inter_dist
        self.pocket_residue_dist = pocket_residue_dist
        self.num_process    = num_process
        self.rebuild        = rebuild
        self.type_map       = utils.get_type_map()
        self.graph_dic_path = os.path.join(cache_dir, 'graph_dic')

        os.makedirs(self.graph_dic_path, exist_ok=True)

        self._pre_process(csv_path)

    # ------------------------------------------------------------------
    def _pre_process(self, csv_path: str):
        keys_bin = os.path.join(self.cache_dir, 'keys.bin')
        lab_bin  = os.path.join(self.cache_dir, 'labels.bin')

        # fast path: index already built, graphs on disk
        if not self.rebuild and os.path.exists(keys_bin):
            print('[IGNDataset] Index found — loading keys & labels …')
            with open(keys_bin, 'rb') as f: self.keys   = pickle.load(f)
            with open(lab_bin,  'rb') as f: self.labels = pickle.load(f)
            print(f'[IGNDataset] {len(self.keys)} complexes indexed.')
            return

        # load CSV
        df = pd.read_csv(csv_path)
        for col in ('label', 'complex_path'):
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: '{col}'")

        if 'compound_id' in df.columns and 'kinase' in df.columns:
            df['pair_id'] = df['kinase'].astype(str) + '_' + df['compound_id'].astype(str)
        else:
            df['pair_id'] = df['complex_path'].apply(
                lambda p: os.path.splitext(os.path.basename(p))[0]
            )

        # pre-filter missing parquets
        mask  = df['complex_path'].apply(os.path.exists)
        n_skip = (~mask).sum()
        if n_skip:
            print(f'[IGNDataset] {n_skip} entries skipped (parquet not found)')
        df = df[mask].reset_index(drop=True)

        if df.empty:
            raise RuntimeError('No valid parquet files found')

        pair_ids_v      = df['pair_id'].tolist()
        labels_v        = df['label'].tolist()
        parquet_paths_v = df['complex_path'].tolist()

        # optionally wipe existing pkls for a clean rebuild
        if self.rebuild:
            import shutil
            shutil.rmtree(self.graph_dic_path)
            os.makedirs(self.graph_dic_path)

        print(f'[IGNDataset] Building graphs for {len(pair_ids_v)} complexes …')

        pool = multiprocessing.Pool(self.num_process)
        pool.starmap(
            build_graph_from_parquet,
            zip(
                parquet_paths_v,
                pair_ids_v,
                labels_v,
                repeat(self.graph_dic_path, len(pair_ids_v)),
                repeat(self.type_map,       len(pair_ids_v)),
                repeat(self.inter_dist,     len(pair_ids_v)),
                repeat(self.pocket_residue_dist, len(pair_ids_v)),
            ),
        )
        pool.close()
        pool.join()

        # collect only successfully built complexes
        self.keys   = []
        self.labels = []
        n_fail = 0
        for pid, lbl in zip(pair_ids_v, labels_v):
            if os.path.exists(os.path.join(self.graph_dic_path, pid + '.pkl')):
                self.keys.append(pid)
                self.labels.append(lbl)
            else:
                n_fail += 1

        print(f'[IGNDataset] {len(self.keys)} graphs OK, {n_fail} failed.')

        # persist index only (no g.bin — graphs stay as individual .pkl files)
        with open(keys_bin, 'wb') as f: pickle.dump(self.keys,   f)
        with open(lab_bin,  'wb') as f: pickle.dump(self.labels, f)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        pkl_path = os.path.join(self.graph_dic_path, key + '.pkl')
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        return (d['g'], d['g3'], torch.tensor(d['label'], dtype=torch.float), d['key'])
    '''
    def __getitem__(self, idx):
        """Load one complex from disk. Called by DataLoader workers per batch."""
        key     = self.keys[idx]
        pkl_path = os.path.join(self.graph_dic_path, key + '.pkl')
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        return (
            d['g'],
            d['g3'],
            torch.tensor(d['label'], dtype=torch.float),
            d['key'],
        )
    '''
    