"""
Extract pocket N/CA/C/O coordinates from one parquet per UniProt and save
as `{pocket_dir}/{uid}.json` in the format KDBNet's `featurize_protein_graph`
expects:

    {
        "name": "<uid>",
        "seq": "KVIGK...",          # pocket sequence (matches pocket_info.csv)
        "coords": list of [N, CA, C, O] triples per residue,
                  shape: [n_res][4][3]
    }

The pocket structure is the same across all ligands for a given UniProt
(only the ligand pose changes), so we only need to extract from ONE parquet
per UniProt. We pick the first parquet listed in train.csv for each kinase.

Critical: we order residues to match `residue_indices` in pocket_info.csv,
so that pocket residues, sequence, and ESM-sliced embeddings are all aligned.

Usage
-----
    python 02_extract_pocket_coords.py \
        --train_csv train.csv \
        --pocket_info pocket_info.csv \
        --pocket_dir data/pocket_coords
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# 3-letter -> 1-letter amino acid codes (20 standard + selenocysteine/pyrrolysine
# fall back to X via .get(default='X')).
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common alternate residue names in MD/Maestro outputs:
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}

BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def parse_residue_indices(s):
    """Parse the stringified list in pocket_info.csv -> List[int]."""
    if isinstance(s, list):
        return s
    return ast.literal_eval(s)


def extract_pocket_from_parquet(parquet_path, residue_indices, expected_seq):
    """
    Build the KDBNet-style pocket dict from one parquet.

    Returns
    -------
    coords_per_res : list of [N, CA, C, O] xyz triples, length = n_pocket_residues
    seq_observed   : 1-letter sequence as observed in the parquet (residues kept)
    kept_indices   : list[int] - residue numbers actually kept
    dropped        : list[(residue_number, reason)]
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["is_ligand"] == 0].copy()

    # Group by (chain_id, residue_number, insertion). Pockets in your data
    # may span multiple chains; we keep all and rely on residue_indices to
    # disambiguate. If residue numbers collide across chains, you'll need a
    # (chain, resnum) key in pocket_info.csv -- ping me if so.
    df["insertion"] = df["insertion"].fillna("").astype(str)

    coords_per_res = []
    seq_chars = []
    kept = []
    dropped = []

    # Build a lookup: residue_number -> sub-dataframe
    by_resnum = {rn: g for rn, g in df.groupby("residue_number")}

    for resnum in residue_indices:
        if resnum not in by_resnum:
            dropped.append((resnum, "missing_in_parquet"))
            continue
        g = by_resnum[resnum]
        # If multiple chains/insertions match, take the first (most common case
        # for a cropped pocket from a single chain).
        if g["chain_id"].nunique() > 1 or g["insertion"].nunique() > 1:
            # Prefer chain A, no insertion
            sub = g[(g["chain_id"] == g["chain_id"].iloc[0]) &
                    (g["insertion"] == g["insertion"].iloc[0])]
        else:
            sub = g

        resname = sub["residue_name"].iloc[0]
        one = THREE_TO_ONE.get(resname, "X")

        # Pull N, CA, C, O coords
        coords_4 = []
        ok = True
        for atom in BACKBONE_ATOMS:
            row = sub[sub["atom_name"] == atom]
            if len(row) == 0:
                ok = False
                dropped.append((resnum, f"missing_{atom}"))
                break
            xyz = row[["x_coord", "y_coord", "z_coord"]].iloc[0].tolist()
            coords_4.append(xyz)
        if not ok:
            continue

        coords_per_res.append(coords_4)
        seq_chars.append(one)
        kept.append(resnum)

    seq_observed = "".join(seq_chars)

    # Sanity check vs. expected_seq from pocket_info.csv. If we dropped any
    # residues for missing backbone atoms, expected_seq will be longer.
    # Verify the kept subset matches the corresponding subset of expected_seq.
    full_index_map = {rn: i for i, rn in enumerate(residue_indices)}
    expected_kept = "".join(expected_seq[full_index_map[rn]] for rn in kept)
    if expected_kept != seq_observed:
        # Soft warning; could happen if Maestro renamed HIS variants etc.
        # We keep the parquet-observed sequence as ground truth for the graph,
        # but flag the mismatch for inspection.
        mismatches = [
            (rn, expected_seq[full_index_map[rn]], seq_observed[i])
            for i, rn in enumerate(kept)
            if expected_seq[full_index_map[rn]] != seq_observed[i]
        ]
        raise ValueError(
            f"sequence mismatch in {parquet_path.name}: {len(mismatches)} residues "
            f"differ. First 3: {mismatches[:3]}"
        )

    return coords_per_res, seq_observed, kept, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True,
                        help="CSV with columns: compound_id, kinase, label, complex_path")
    parser.add_argument("--pocket_info", required=True,
                        help="CSV with columns: uniprot_id, pocket_sequence, residue_indices, ...")
    parser.add_argument("--pocket_dir", required=True,
                        help="Output directory for {uid}.json files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    Path(args.pocket_dir).mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    pocket_info = pd.read_csv(args.pocket_info).set_index("uniprot_id")

    # Pick one parquet per kinase (first one seen)
    one_parquet_per_uid = train.drop_duplicates("kinase").set_index("kinase")["complex_path"].to_dict()

    failed = []
    for uid, parquet_path in tqdm(one_parquet_per_uid.items(), desc="pocket"):
        out_path = Path(args.pocket_dir) / f"{uid}.json"
        if out_path.exists() and not args.overwrite:
            continue
        if uid not in pocket_info.index:
            failed.append((uid, "missing_in_pocket_info"))
            continue

        row = pocket_info.loc[uid]
        residue_indices = parse_residue_indices(row["residue_indices"])
        expected_seq = row["pocket_sequence"]

        try:
            coords, seq, kept, dropped = extract_pocket_from_parquet(
                Path(parquet_path), residue_indices, expected_seq)
        except Exception as e:
            failed.append((uid, str(e)))
            continue

        if len(coords) < 5:
            failed.append((uid, f"only_{len(coords)}_residues_kept"))
            continue

        entry = {
            "name": uid,
            "uniprot_id": uid,
            "seq": seq,
            "coords": coords,                 # list of [[N], [CA], [C], [O]] per residue
            "kept_residue_indices": kept,     # 1-indexed UniProt positions, aligned with coords/seq
            "dropped": dropped,               # for debugging
        }
        with open(out_path, "w") as f:
            json.dump(entry, f)

    if failed:
        pd.DataFrame(failed, columns=["uniprot_id", "error"]).to_csv(
            Path(args.pocket_dir) / "_failed.csv", index=False)
        print(f"[warn] {len(failed)} failures logged to {args.pocket_dir}/_failed.csv")
    print("[done]")


if __name__ == "__main__":
    main()