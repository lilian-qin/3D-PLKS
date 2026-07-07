import pandas as pd
import numpy as np
import os
from rdkit import Chem
import networkx as nx
import torch
from utils import *


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
             'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
             'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index


def seq_cat(prot, seq_dict, max_seq_len):
    x = np.zeros(max_seq_len, dtype=np.int16)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)
    return x


# ============================================================
# Config
# ============================================================
MAX_SEQ_LEN = 108  # adjust based on your sequence length stats

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}

# ---- Check sequence lengths ----
print("Checking sequence lengths across all splits...")
for split in ['test_pocket_gta']:
    df = pd.read_csv(f'data_random_split/{split}.csv')
    lens_A = df['sequence_A'].str.len()
    lens_B = df['sequence_B'].str.len()
    all_lens = pd.concat([lens_A, lens_B])
    truncated = (all_lens > MAX_SEQ_LEN).sum()
    print(f"  {split}: {len(df)} samples, "
          f"max_len={all_lens.max()}, median={all_lens.median():.0f}, "
          f"truncated={truncated} ({truncated/len(all_lens)*100:.1f}%)")

# ---- Build SMILES graphs ----
splits = ['test_pocket_gta']
compound_iso_smiles = set()
for split in splits:
    df = pd.read_csv(f'data_random_split/{split}.csv')
    compound_iso_smiles.update(df['SMILES'].unique())

print(f"\nBuilding graphs for {len(compound_iso_smiles)} unique SMILES...")
smile_graph = {}
failed = []
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    if g is not None:
        smile_graph[smile] = g
    else:
        failed.append(smile)
if failed:
    print(f"  WARNING: {len(failed)} SMILES failed to parse")
print(f"Built {len(smile_graph)} graphs")

# ---- Process each split ----
for split in splits:
    processed_file = f'data_random_split/processed/selectivity_{split}.pt'

    if os.path.isfile(processed_file):
        print(f"\n{processed_file} already exists, skipping")
        print(f"  (delete it to regenerate)")
        continue

    df = pd.read_csv(f'data_random_split/{split}.csv')
    df = df[df['SMILES'].isin(smile_graph)].reset_index(drop=True)
    print(f"\nProcessing {split}: {len(df)} samples (max_seq_len={MAX_SEQ_LEN})")

    xd = np.asarray(df['SMILES'].tolist())
    xt_A = np.asarray([seq_cat(seq, seq_dict, MAX_SEQ_LEN) for seq in df['sequence_A']])
    xt_B = np.asarray([seq_cat(seq, seq_dict, MAX_SEQ_LEN) for seq in df['sequence_B']])
    y = np.asarray(df['label'].tolist())

    # IDs for inference output
    compound_ids = np.asarray(df['compound_id'].tolist())
    kinase_A_ids = np.asarray(df['kinase_A'].tolist())
    kinase_B_ids = np.asarray(df['kinase_B'].tolist())

    dataset = SiameseTestbedDataset(
        root='data_random_split',
        dataset=f'selectivity_{split}',
        xd=xd,
        xt_A=xt_A,
        xt_B=xt_B,
        y=y,
        smile_graph=smile_graph,
        compound_ids=compound_ids,
        kinase_A_ids=kinase_A_ids,
        kinase_B_ids=kinase_B_ids,
        max_seq_len=MAX_SEQ_LEN,
    )
    print(f"  Created {processed_file}")

print("\nDone!")
print(f"MAX_SEQ_LEN = {MAX_SEQ_LEN}")
print("Remember to update model Conv1d layer accordingly:")
print(f"  self.conv_xt_1 = nn.Conv1d(in_channels={MAX_SEQ_LEN}, out_channels=n_filters, kernel_size=8)")
print(f"  self.fc1_xt = nn.Linear(32 * {MAX_SEQ_LEN - 7}, output_dim)")