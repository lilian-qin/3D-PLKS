"""
Diagnostic + fix script for kinases that fail alignment.

For each failed kinase:
  1. Fetch canonical sequence from UniProt REST API.
  2. Compare to the `full_sequence` in pocket_info.csv.
  3. Try aligning the parquet pocket sequence to the canonical UniProt sequence.
  4. Report which kinases need an updated full_sequence.

If you want to actually update pocket_info.csv with the fetched sequences,
pass --apply. Otherwise it just prints a report.

Note: this script uses urllib so it has no dependencies beyond the standard
library + pandas. Make sure the cluster has outbound internet access; if not,
fetch the sequences locally and put them in a TSV (uniprot_id, sequence) and
use --canonical_tsv instead of --fetch.

Usage
-----
    # Fetch and report
    python scripts/05_fix_failed_alignments.py \
        --failed_csv data/kdbnet/proteins/_failed.csv \
        --pocket_info data/full_pocket_sequences.csv \
        --pocket_dir data/pocket_coords \
        --fetch

    # Or with a pre-fetched TSV
    python scripts/05_fix_failed_alignments.py \
        --failed_csv data/kdbnet/proteins/_failed.csv \
        --pocket_info data/full_pocket_sequences.csv \
        --pocket_dir data/pocket_coords \
        --canonical_tsv canonical_uniprot_seqs.tsv

    # Apply the fix to a new CSV
    python scripts/05_fix_failed_alignments.py \
        --failed_csv data/kdbnet/proteins/_failed.csv \
        --pocket_info data/full_pocket_sequences.csv \
        --pocket_dir data/pocket_coords \
        --fetch \
        --out_csv data/full_pocket_sequences_fixed.csv
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


def fetch_uniprot(uid, retries=3, sleep=0.3):
    """Fetch canonical FASTA from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    last_err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                txt = resp.read().decode()
            lines = txt.splitlines()
            seq = "".join(line for line in lines if not line.startswith(">"))
            return seq
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(sleep)
    raise RuntimeError(f"failed to fetch {uid}: {last_err}")


def hamming_with_offset(target, full, offset):
    """Mismatches when slicing full at indices+offset for indices implied by target length."""
    L = len(target)
    if offset < 0 or offset + L > len(full):
        return float("inf")
    return sum(c1 != c2 for c1, c2 in zip(target, full[offset:offset + L]))


def best_substring_alignment(target, full):
    """Return (best_start, n_mismatch). Linear scan."""
    L = len(target)
    if L > len(full):
        return None, float("inf")
    best = (None, float("inf"))
    for s in range(len(full) - L + 1):
        nm = sum(c1 != c2 for c1, c2 in zip(target, full[s:s + L]))
        if nm < best[1]:
            best = (s, nm)
            if nm == 0:
                break
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed_csv", required=True)
    parser.add_argument("--pocket_info", required=True)
    parser.add_argument("--pocket_dir", required=True)
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch sequences from UniProt REST API")
    parser.add_argument("--canonical_tsv", default=None,
                        help="Alternative: TSV with uniprot_id\\tsequence to use "
                             "instead of fetching")
    parser.add_argument("--out_csv", default=None,
                        help="If given, write a corrected pocket_info CSV here "
                             "with full_sequence updated for the failed kinases.")
    args = parser.parse_args()

    failed = pd.read_csv(args.failed_csv)
    info = pd.read_csv(args.pocket_info).set_index("uniprot_id")

    # Skip the file_missing failures (those need step 01 fixes, not sequence fixes)
    failed = failed[~failed["error"].str.contains("file_missing", na=False)]
    failed = failed[~failed["error"].str.contains("residue index out of range", na=False)]

    # Load canonical sequences
    if args.canonical_tsv:
        canon = pd.read_csv(args.canonical_tsv, sep="\t",
                            names=["uniprot_id", "sequence"]).set_index("uniprot_id")
        canon_dict = canon["sequence"].to_dict()
    elif args.fetch:
        canon_dict = {}
        for uid in failed["uniprot_id"]:
            try:
                canon_dict[uid] = fetch_uniprot(uid)
                print(f"  fetched {uid}: {len(canon_dict[uid])} aa")
            except Exception as e:
                print(f"  FETCH FAILED {uid}: {e}")
    else:
        raise SystemExit("Pass either --fetch or --canonical_tsv")

    # Compare
    rows = []
    for uid in failed["uniprot_id"]:
        if uid not in canon_dict:
            rows.append({"uniprot_id": uid, "status": "no_canonical_fetched"})
            continue
        canonical = canon_dict[uid]
        current_seq = info.loc[uid, "full_sequence"] if uid in info.index else ""

        # Load parquet pocket sequence from the pocket JSON
        with open(Path(args.pocket_dir) / f"{uid}.json") as f:
            pocket_seq = json.load(f)["seq"]

        # Try aligning pocket to canonical
        start, nm = best_substring_alignment(pocket_seq, canonical)
        L = len(pocket_seq)

        rows.append({
            "uniprot_id": uid,
            "current_len": len(current_seq),
            "canonical_len": len(canonical),
            "seq_changed": current_seq != canonical,
            "pocket_aligns_to_canonical": start is not None and nm <= int(0.10 * L),
            "best_start_in_canonical": start,
            "n_mismatch_canonical": nm,
            "pocket_len": L,
        })

    df = pd.DataFrame(rows)
    print("\n[report]")
    print(df.to_string())
    print()
    print(f"Kinases where canonical UniProt sequence DIFFERS from pocket_info: "
          f"{df['seq_changed'].sum()}/{len(df)}")
    print(f"Kinases where pocket aligns cleanly to canonical: "
          f"{df['pocket_aligns_to_canonical'].sum()}/{len(df)}")

    if args.out_csv:
        new_info = info.reset_index().copy()
        n_updated = 0
        for _, r in df.iterrows():
            uid = r["uniprot_id"]
            if uid in canon_dict and r["pocket_aligns_to_canonical"]:
                mask = new_info["uniprot_id"] == uid
                new_info.loc[mask, "full_sequence"] = canon_dict[uid]
                n_updated += 1
        new_info.to_csv(args.out_csv, index=False)
        print(f"\n[write] updated {n_updated} kinases, saved to {args.out_csv}")


if __name__ == "__main__":
    main()