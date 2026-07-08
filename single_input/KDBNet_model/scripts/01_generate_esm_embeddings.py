"""
Generate ESM2 per-residue embeddings for full UniProt sequences.

Reads `pocket_info.csv` (must have `uniprot_id` and `full_sequence` columns),
runs ESM2 (esm2_t33_650M_UR50D, 1280-d) once per kinase, and saves
{esm_dir}/{uid}.pt  -- a tensor of shape [L_full, 1280].

This is a one-time operation. Cost: ~30 min on one H100 for ~500 kinases.

Usage
-----
    python 01_generate_esm_embeddings.py \
        --pocket_info pocket_info.csv \
        --esm_dir data/esm2_full \
        --batch_size 4 \
        --device cuda

Notes
-----
- Embeddings are 0-indexed: esm[i] corresponds to the (i+1)-th residue
  in the 1-indexed UniProt sequence. Slicing with `residue_indices` from
  `pocket_info.csv` (which uses 1-indexed PDB/UniProt numbering) requires
  subtracting 1. This is handled in `03_build_kdbnet_dataset.py`.
- Skips kinases whose .pt already exists (resumable).
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def load_esm2(device):
    """Load esm2_t33_650M_UR50D (1280-d, matches KDBNet's d_pretrained_emb)."""
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


@torch.no_grad()
def embed_sequence(model, batch_converter, uid, sequence, device, repr_layer=33):
    """
    Run ESM2 on a single sequence and return per-residue embeddings.

    Returns
    -------
    emb : torch.FloatTensor of shape [L, 1280] on CPU
    """
    data = [(uid, sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    out = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    # Strip BOS (idx 0) and EOS (idx L+1) -> keep [1 : L+1]
    emb = out["representations"][repr_layer][0, 1 : len(sequence) + 1].cpu()
    assert emb.shape == (len(sequence), 1280), f"Got {emb.shape} for {uid}"
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket_info", required=True,
                        help="CSV with columns: uniprot_id, full_sequence, ...")
    parser.add_argument("--esm_dir", required=True,
                        help="Output directory for {uid}.pt files")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_length", type=int, default=2046,
                        help="Skip sequences longer than this (ESM2 limit ~1024 "
                             "for some variants; 650M handles up to ~2046).")
    args = parser.parse_args()

    Path(args.esm_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pocket_info)
    df = df[["uniprot_id", "full_sequence"]].drop_duplicates("uniprot_id")
    print(f"[info] {len(df)} unique kinases in {args.pocket_info}")

    # Filter already-done
    todo = []
    for _, row in df.iterrows():
        uid = row["uniprot_id"]
        out_path = Path(args.esm_dir) / f"{uid}.pt"
        if out_path.exists():
            continue
        seq = row["full_sequence"]
        if len(seq) > args.max_length:
            print(f"[warn] {uid} sequence length {len(seq)} > {args.max_length}, skipping")
            continue
        todo.append((uid, seq))
    print(f"[info] {len(todo)} kinases to embed (rest already cached)")

    if not todo:
        return

    model, batch_converter = load_esm2(args.device)
    print(f"[info] ESM2 loaded on {args.device}")

    failed = []
    for uid, seq in tqdm(todo, desc="ESM2"):
        try:
            emb = embed_sequence(model, batch_converter, uid, seq, args.device)
            torch.save(emb, Path(args.esm_dir) / f"{uid}.pt")
        except Exception as e:
            print(f"[error] {uid} ({len(seq)} aa): {e}")
            failed.append((uid, str(e)))
            torch.cuda.empty_cache()

    if failed:
        fail_df = pd.DataFrame(failed, columns=["uniprot_id", "error"])
        fail_df.to_csv(Path(args.esm_dir) / "_failed.csv", index=False)
        print(f"[warn] {len(failed)} failures logged to {args.esm_dir}/_failed.csv")
    print("[done]")


if __name__ == "__main__":
    main()