"""
Build cached PyG drug graphs from SDF files at
    docking_dataset/{uid}/{lig_name}.sdf

Output:
    drugs/{lig_name}.pt   (a torch_geometric Data object)

Note: KDBNet's featurize_drug uses radius_graph (4.5 A) on 3D coords,
not RDKit bonds. So as long as the SDF gives RDKit a valid Mol with 3D
coords, the docking pose is what becomes the ligand graph -- not the
SDF's reference conformer (because the parquet was generated from the
SDF, the SDF coords ARE the docked pose).

Ligand caching is keyed by lig_name (compound_id), since the same compound
can appear under multiple kinases. This dedup is important: typical kinase
panels have ~50k unique compounds but ~10x more pairs.

Usage
-----
    python 04_build_drug_graphs.py \
        --train_csv train.csv \
        --sdf_root docking_dataset \
        --out_dir data/kdbnet/drugs
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from kdbnet.mol_graph import featurize_drug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--sdf_root", required=True,
                        help="Root containing {uid}/{lig_name}.sdf")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    # We need (compound_id, kinase) so we can find the SDF path.
    # If the same compound docks to multiple kinases we just pick the first SDF
    # we find -- the ligand's chemistry is the same; coords differ per pose
    # but featurize_drug encodes 3D, so we want a consistent canonical pose.
    # Convention: use the SDF under the FIRST kinase that has the compound.
    pairs = df[["compound_id", "kinase"]].drop_duplicates("compound_id")

    failed = []
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="drug graphs"):
        lig = str(row["compound_id"])
        uid = row["kinase"]
        out_path = Path(args.out_dir) / f"{lig}.pt"
        if out_path.exists() and not args.overwrite:
            continue
        sdf_path = Path(args.sdf_root) / uid / f"{lig}.sdf"
        if not sdf_path.exists():
            failed.append((lig, f"missing_sdf:{sdf_path}"))
            continue
        try:
            data = featurize_drug(str(sdf_path), name=lig)
            torch.save(data, out_path)
        except Exception as e:
            failed.append((lig, str(e)))

    if failed:
        pd.DataFrame(failed, columns=["compound_id", "error"]).to_csv(
            Path(args.out_dir) / "_failed.csv", index=False)
        print(f"[warn] {len(failed)} failures logged to {args.out_dir}/_failed.csv")
    print("[done]")


if __name__ == "__main__":
    main()