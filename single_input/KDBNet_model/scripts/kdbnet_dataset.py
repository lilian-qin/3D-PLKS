"""
Runtime PyG dataset for KDBNet, adapted to your parquet/SDF/CSV layout.

This dataset is API-compatible with KDBNet's `DTA` class:
    item = dataset[i]
    item["drug"]    -- torch_geometric.data.Data (cached drug graph)
    item["protein"] -- torch_geometric.data.Data (cached protein graph)
    item["y"]       -- float (pIC50 / label)

Cached graphs are loaded lazily on __getitem__ to keep memory low. If you
have plenty of RAM and want maximum throughput, set `preload=True`.

The collate_fn batches drug and protein independently (KDBNet has no
protein-ligand interaction edges; the two towers are merged only at the
top MLP).

Usage
-----
    from kdbnet_dataset import KDBNetCachedDataset, kdbnet_collate

    train = KDBNetCachedDataset(
        pairs_csv="train.csv",
        prot_dir="data/kdbnet/proteins",
        drug_dir="data/kdbnet/drugs",
        label_col="label",
    )
    loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True,
        collate_fn=kdbnet_collate, num_workers=4)

    for batch in loader:
        out = model(batch["drug"], batch["protein"])
        loss = mse(out.squeeze(-1), batch["y"])
"""
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class KDBNetCachedDataset(Dataset):
    """
    Dataset over (compound_id, kinase, label) triples.

    Parameters
    ----------
    pairs_csv : str
        CSV with columns: compound_id, kinase, label   (your train.csv layout)
    prot_dir : str
        Directory containing {uid}.pt cached protein graphs.
    drug_dir : str
        Directory containing {lig_name}.pt cached drug graphs.
    label_col : str, default 'label'
        Column in pairs_csv to use as the regression target.
    drop_missing : bool, default True
        Drop rows where the protein or drug .pt is missing on disk.
    preload : bool, default False
        If True, load every protein and drug graph into RAM up front.
        Recommended only if dataset fits comfortably (~10s GB).
    """

    def __init__(self, pairs_csv, prot_dir, drug_dir,
                 label_col="label", drop_missing=True, preload=False):
        self.prot_dir = Path(prot_dir)
        self.drug_dir = Path(drug_dir)
        self.label_col = label_col

        df = pd.read_csv(pairs_csv)
        df["compound_id"] = df["compound_id"].astype(str)
        df["kinase"] = df["kinase"].astype(str)

        if drop_missing:
            n_before = len(df)
            df = df[df.apply(
                lambda r: (self.prot_dir / f"{r['kinase']}.pt").exists()
                          and (self.drug_dir / f"{r['compound_id']}.pt").exists(),
                axis=1
            )].reset_index(drop=True)
            n_after = len(df)
            if n_after < n_before:
                print(f"[KDBNetCachedDataset] dropped {n_before - n_after} pairs "
                      f"with missing cached graphs")
        self.df = df

        self._prot_cache = None
        self._drug_cache = None
        if preload:
            self._preload()

    def _preload(self):
        prot_uids = sorted(self.df["kinase"].unique())
        drug_lids = sorted(self.df["compound_id"].unique())
        print(f"[KDBNetCachedDataset] preloading {len(prot_uids)} proteins, "
              f"{len(drug_lids)} drugs")
        self._prot_cache = {u: torch.load(self.prot_dir / f"{u}.pt", weights_only=False)
                            for u in prot_uids}
        self._drug_cache = {d: torch.load(self.drug_dir / f"{d}.pt", weights_only=False)
                            for d in drug_lids}

    def _load_prot(self, uid):
        if self._prot_cache is not None:
            return self._prot_cache[uid]
        return torch.load(self.prot_dir / f"{uid}.pt", weights_only=False)

    def _load_drug(self, lig):
        if self._drug_cache is not None:
            return self._drug_cache[lig]
        return torch.load(self.drug_dir / f"{lig}.pt", weights_only=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prot = self._load_prot(row["kinase"])
        drug = self._load_drug(row["compound_id"])
        y = float(row[self.label_col])
        return {
            "drug": drug,
            "protein": prot,
            "y": y,
            "drug_name": row["compound_id"],
            "protein_name": row["kinase"],
        }


def kdbnet_collate(batch):
    """
    Collate a list of dataset items into batched PyG graphs.

    Returns a dict with:
        'drug':    torch_geometric.data.Batch
        'protein': torch_geometric.data.Batch
        'y':       FloatTensor [B]
    """
    drugs = [item["drug"] for item in batch]
    prots = [item["protein"] for item in batch]
    ys = torch.tensor([item["y"] for item in batch], dtype=torch.float32)
    return {
        "drug": Batch.from_data_list(drugs),
        "protein": Batch.from_data_list(prots),
        "y": ys,
    }