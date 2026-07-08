"""
Build cached PyG protein graphs from pocket coords + ESM embeddings,
with robust handling of indexing mismatches.

This is a more forgiving version of 03_build_protein_graphs.py. It:

  1. Auto-detects 1-vs-0-indexed `residue_indices` per kinase by trying
     both and picking whichever gives a better match to the parquet sequence.
  2. Tolerates up to `--mismatch_tol` fraction of single-residue substitutions
     (default 5%) -- these are usually isoform/variant differences and ESM
     embeddings of similar residues remain informative.
  3. Falls back to substring realignment: if neither indexing scheme matches,
     it searches for the pocket sequence as a substring of the full sequence
     (allowing up to `--mismatch_tol` mismatches) and recomputes correct
     residue_indices.
  4. Logs which fallback was used per kinase.

Usage
-----
    python scripts/03b_build_protein_graphs_robust.py \
        --pocket_dir data/pocket_coords \
        --esm_dir data/esm2_full \
        --pocket_info data/full_pocket_sequences.csv \
        --out_dir data/kdbnet/proteins \
        --mismatch_tol 0.05
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from kdbnet.pdb_graph import featurize_protein_graph


def hamming(a, b):
    """Number of positions where strings differ (assumes same length)."""
    assert len(a) == len(b)
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def try_indexing(full_seq, indices, target_seq, offset):
    """
    Try slicing full_seq at `indices + offset` and compare to target_seq.

    Returns
    -------
    sliced : str or None        (None if any index out of range)
    n_mismatch : int or inf
    """
    if any(i + offset < 0 or i + offset >= len(full_seq) for i in indices):
        return None, float("inf")
    sliced = "".join(full_seq[i + offset] for i in indices)
    return sliced, hamming(sliced, target_seq)


def find_substring_realignment(full_seq, target_seq, max_mismatch):
    """
    Find a contiguous window in full_seq matching target_seq with <=max_mismatch
    substitutions.

    Returns
    -------
    start_position : int or None  (0-indexed start in full_seq, or None if no match)
    n_mismatch : int
    """
    L = len(target_seq)
    best = (None, len(target_seq) + 1)
    for start in range(len(full_seq) - L + 1):
        window = full_seq[start : start + L]
        nm = hamming(window, target_seq)
        if nm < best[1]:
            best = (start, nm)
            if nm == 0:
                break
    if best[1] <= max_mismatch:
        return best
    return (None, best[1])


def resolve_indices(full_seq, parquet_seq, residue_indices, mismatch_tol):
    """
    Determine the correct residue_indices alignment.

    Returns
    -------
    esm_idx_0based : List[int]   indices into ESM tensor (0-indexed)
    method : str                 description of which fallback was used
    n_mismatch : int             remaining mismatch count
    """
    L = len(parquet_seq)
    max_mm = int(np.ceil(mismatch_tol * L))

    # Try 1-indexed (offset = -1)
    sliced, nm = try_indexing(full_seq, residue_indices, parquet_seq, offset=-1)
    if sliced is not None and nm <= max_mm:
        return ([i - 1 for i in residue_indices],
                f"1-indexed (mismatches={nm}/{L})", nm)

    # Try 0-indexed (offset = 0)
    sliced0, nm0 = try_indexing(full_seq, residue_indices, parquet_seq, offset=0)
    if sliced0 is not None and nm0 <= max_mm:
        return ([i for i in residue_indices],
                f"0-indexed (mismatches={nm0}/{L})", nm0)

    # Substring realignment
    start, nm_sub = find_substring_realignment(full_seq, parquet_seq, max_mm)
    if start is not None:
        return (list(range(start, start + L)),
                f"realigned to position {start} (mismatches={nm_sub}/{L})", nm_sub)

    # All failed; return the best of the three with full diagnostic
    best_nm = min(nm, nm0, nm_sub)
    raise ValueError(
        f"could not align pocket to full sequence within tolerance "
        f"{max_mm}/{L}. best mismatch counts: "
        f"1-indexed={nm}, 0-indexed={nm0}, substring={nm_sub}"
    )


def build_one_protein(uid, pocket_dir, esm_dir, full_seq, mismatch_tol):
    pocket_path = Path(pocket_dir) / f"{uid}.json"
    esm_path = Path(esm_dir) / f"{uid}.pt"

    with open(pocket_path) as f:
        entry = json.load(f)

    parquet_seq = entry["seq"]
    coords = entry["coords"]
    raw_indices = entry["kept_residue_indices"]

    # Validate ESM
    esm_full = torch.load(esm_path, weights_only=True)
    if esm_full.shape[0] != len(full_seq):
        raise ValueError(
            f"ESM length {esm_full.shape[0]} != full_seq length {len(full_seq)}")

    # Resolve indexing
    esm_idx, method, nm = resolve_indices(
        full_seq, parquet_seq, raw_indices, mismatch_tol)

    seq_emb = esm_full[torch.tensor(esm_idx, dtype=torch.long)].clone()
    assert seq_emb.shape == (len(parquet_seq), 1280)

    protein_dict = {
        "name": uid,
        "seq": parquet_seq,
        "coords": coords,
    }
    data = featurize_protein_graph(protein_dict, name=uid)
    data.seq_emb = seq_emb
    return data, method, nm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket_dir", required=True)
    parser.add_argument("--esm_dir", required=True)
    parser.add_argument("--pocket_info", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mismatch_tol", type=float, default=0.05,
                        help="Max fraction of residues allowed to mismatch between "
                             "parquet pocket sequence and the corresponding slice "
                             "of the full UniProt sequence (default 0.05 = 5%%).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    info = pd.read_csv(args.pocket_info).set_index("uniprot_id")

    failed = []
    succeeded = []
    uids = sorted({p.stem for p in Path(args.pocket_dir).glob("*.json")})
    for uid in tqdm(uids, desc="protein graphs (robust)"):
        out_path = Path(args.out_dir) / f"{uid}.pt"
        if out_path.exists() and not args.overwrite:
            continue
        if uid not in info.index:
            failed.append((uid, "missing_in_pocket_info"))
            continue

        try:
            data, method, nm = build_one_protein(
                uid, args.pocket_dir, args.esm_dir,
                full_seq=info.loc[uid, "full_sequence"],
                mismatch_tol=args.mismatch_tol)
            torch.save(data, out_path)
            succeeded.append((uid, method, nm))
        except FileNotFoundError as e:
            failed.append((uid, f"file_missing: {e}"))
        except Exception as e:
            failed.append((uid, str(e)))

    pd.DataFrame(succeeded, columns=["uniprot_id", "method", "n_mismatch"]).to_csv(
        Path(args.out_dir) / "_alignment_log.csv", index=False)

    if failed:
        pd.DataFrame(failed, columns=["uniprot_id", "error"]).to_csv(
            Path(args.out_dir) / "_failed.csv", index=False)
        print(f"[warn] {len(failed)} failures logged to {args.out_dir}/_failed.csv")
    print(f"[done] {len(succeeded)} succeeded, {len(failed)} failed")

    # Summary of which methods were used
    if succeeded:
        df = pd.DataFrame(succeeded, columns=["uniprot_id", "method", "n_mismatch"])
        df["method_kind"] = df["method"].str.split(" ").str[0]
        print("\n[summary]")
        print(df["method_kind"].value_counts().to_string())
        print(f"\nmismatch distribution: {df['n_mismatch'].describe().to_string()}")


if __name__ == "__main__":
    main()