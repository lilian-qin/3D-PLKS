
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from kdbnet.model import DTAModel

# Make sure scripts/ is on path for kdbnet_dataset import.
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from kdbnet_dataset import KDBNetCachedDataset, kdbnet_collate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint_path, device):
    """Load a checkpoint. Returns (state_dict, model_kwargs).

    Supports three checkpoint formats:
    - new bundled best_model.pt: dict with keys 'model' and 'model_kwargs'
    - new bundled last_checkpoint.pt: dict with 'model', 'model_kwargs', 'optimizer'
    - legacy plain state_dict (no hyperparameters bundled): falls back to defaults
    """
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
        kwargs = obj.get("model_kwargs", None)
        if kwargs is None and "args" in obj:
            # last_checkpoint.pt may have full args dict but no extracted model_kwargs;
            # pluck the relevant subset.
            a = obj["args"]
            kwargs = {k: a[k] for k in (
                "prot_emb_dim", "prot_gcn_dims", "prot_fc_dims",
                "drug_node_h_dims", "drug_edge_h_dims", "drug_fc_dims",
                "mlp_dims", "mlp_dropout") if k in a}
    else:
        # Legacy plain state_dict
        state = obj
        kwargs = None
    return state, kwargs


def build_model(model_kwargs, device):
    """Reconstruct DTAModel. If model_kwargs is None, use DTAModel defaults."""
    kwargs = model_kwargs or {}
    model = DTAModel(
        prot_emb_dim=kwargs.get("prot_emb_dim", 1280),
        prot_gcn_dims=kwargs.get("prot_gcn_dims", [128, 256, 256]),
        prot_fc_dims=kwargs.get("prot_fc_dims", [1024, 128]),
        drug_node_h_dims=kwargs.get("drug_node_h_dims", [128, 64]),
        drug_edge_h_dims=kwargs.get("drug_edge_h_dims", [32, 1]),
        drug_fc_dims=kwargs.get("drug_fc_dims", [1024, 128]),
        mlp_dims=kwargs.get("mlp_dims", [1024, 512]),
        mlp_dropout=kwargs.get("mlp_dropout", 0.25),
    ).to(device)
    return model


@torch.no_grad()
def predict(model, loader, device):
    """Run forward pass over the entire loader. Returns y_pred + identifiers."""
    model.eval()
    preds, drug_names, prot_names = [], [], []
    has_labels = False
    ys = []
    for batch in loader:
        drug = batch["drug"].to(device)
        prot = batch["protein"].to(device)
        pred = model(drug, prot).squeeze(-1).cpu()
        preds.append(pred)
        drug_names.extend(batch["drug"].name)
        prot_names.extend(batch["protein"].name)
        if "y" in batch and batch["y"] is not None:
            ys.append(batch["y"])
            has_labels = True
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(ys).numpy() if has_labels else None
    return drug_names, prot_names, y_pred, y_true


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "n": int(len(y_true)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "pearson": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan"),
        "spearman": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", required=True,
                        help="CSV with at least compound_id and kinase columns")
    parser.add_argument("--checkpoint", required=True, nargs="+",
                        help="Path to best_model.pt (or multiple, for ensembling). "
                             "Hyperparameters are read from the checkpoint itself "
                             "if it was saved by the new train.py; otherwise "
                             "DTAModel defaults are used.")
    parser.add_argument("--prot_dir", default="data/kdbnet/proteins",
                        help="Cached protein graphs directory")
    parser.add_argument("--drug_dir", default="data/kdbnet/drugs",
                        help="Cached drug graphs directory")
    parser.add_argument("--label_col", default='label',
                        help="If set, treat this column as ground-truth labels "
                             "and compute metrics. Default: don't expect labels.")
    parser.add_argument("--out_csv", required=True,
                        help="Output predictions CSV")
    parser.add_argument("--metrics_json", default=None,
                        help="If --label_col set, write overall metrics here. "
                             "Defaults to {out_csv}.metrics.json")
    parser.add_argument("--per_kinase_csv", default=None,
                        help="If set, write per-kinase metrics here. Requires "
                             "--label_col.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}, ensemble_size={len(args.checkpoint)}")

    # Pre-load all checkpoints' state_dicts and hyperparameters
    states = []
    first_kwargs = None
    for ckpt_path in args.checkpoint:
        state, kwargs = load_checkpoint(ckpt_path, device)
        if first_kwargs is None:
            first_kwargs = kwargs
        elif kwargs != first_kwargs and kwargs is not None:
            print(f"[warn] {ckpt_path} has different hyperparameters than first "
                   "checkpoint; using first checkpoint's architecture")
        states.append((ckpt_path, state))
    if first_kwargs is None:
        print("[warn] checkpoints have no embedded hyperparameters; using DTAModel "
              "defaults. If you trained with non-default hyperparameters, use a "
              "newer checkpoint or pass them manually.")
    else:
        print(f"[setup] architecture: {first_kwargs}")

    # Load input CSV
    df = pd.read_csv(args.pairs_csv)
    df["compound_id"] = df["compound_id"].astype(str)
    df["kinase"] = df["kinase"].astype(str)
    print(f"[data] loaded {len(df)} pairs from {args.pairs_csv}")

    # Build dataset. If a label column exists, route it through; otherwise
    # we add a dummy column so KDBNetCachedDataset doesn't choke.
    label_col = args.label_col
    if label_col is None:
        df = df.copy()
        df["_dummy_label"] = 0.0
        label_col_for_ds = "_dummy_label"
    else:
        label_col_for_ds = label_col

    # Write to a temp file for KDBNetCachedDataset (which reads from a path)
    tmp_csv = Path(args.out_csv).with_suffix(".tmp.csv")
    df.to_csv(tmp_csv, index=False)

    try:
        ds = KDBNetCachedDataset(
            str(tmp_csv), args.prot_dir, args.drug_dir,
            label_col=label_col_for_ds, drop_missing=True)
    finally:
        tmp_csv.unlink(missing_ok=True)

    if len(ds) == 0:
        raise SystemExit(
            "All input pairs were dropped (missing cached graphs). Check that "
            "data/kdbnet/proteins and data/kdbnet/drugs contain .pt files for "
            "the kinases and compounds in your CSV.")
    if len(ds) < len(df):
        print(f"[warn] {len(df) - len(ds)} pairs dropped due to missing cached graphs")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=kdbnet_collate, num_workers=args.num_workers,
                        pin_memory=True)

    # Build model once -- we reuse it across checkpoints in the ensemble loop
    model = build_model(first_kwargs, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params:,} params")

    # Run inference (potentially across multiple checkpoints)
    all_preds = []
    drug_names = prot_names = y_true_arr = None
    for ckpt_path, state in states:
        print(f"[predict] {ckpt_path}")
        model.load_state_dict(state)
        d_names, p_names, y_pred, y_true = predict(model, loader, device)
        if drug_names is None:
            drug_names, prot_names, y_true_arr = d_names, p_names, y_true
        all_preds.append(y_pred)

    all_preds = np.stack(all_preds, axis=0)  # [n_ckpt, n_pairs]
    y_pred_mean = all_preds.mean(axis=0)
    y_pred_std = all_preds.std(axis=0) if all_preds.shape[0] > 1 else None

    # Build output DataFrame
    out_df = pd.DataFrame({
        "compound_id": drug_names,
        "kinase": prot_names,
        "y_pred": y_pred_mean,
    })
    if y_pred_std is not None:
        out_df["y_std"] = y_pred_std
    if y_true_arr is not None and label_col is not None:
        out_df["y_true"] = y_true_arr
        out_df["residual"] = y_pred_mean - y_true_arr
        out_df["abs_error"] = np.abs(out_df["residual"])

    out_df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {len(out_df)} predictions to {args.out_csv}")

    # Metrics + per-kinase breakdown if labels present
    if y_true_arr is not None and label_col is not None:
        metrics = compute_metrics(y_true_arr, y_pred_mean)
        metrics_path = args.metrics_json or (str(args.out_csv) + ".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[metrics] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in metrics.items()))

        if args.per_kinase_csv:
            rows = []
            for k, sub in out_df.groupby("kinase"):
                if len(sub) < 5:
                    rows.append({"kinase": k, "n": len(sub),
                                  "pearson": float("nan"),
                                  "rmse": float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2))),
                                  "mae": float(np.mean(np.abs(sub["y_true"] - sub["y_pred"])))})
                else:
                    m = compute_metrics(sub["y_true"].values, sub["y_pred"].values)
                    rows.append({"kinase": k, "n": m["n"],
                                  "pearson": m["pearson"], "rmse": m["rmse"], "mae": m["mae"]})
            pd.DataFrame(rows).sort_values("n", ascending=False).to_csv(
                args.per_kinase_csv, index=False)
            print(f"[metrics] per-kinase breakdown -> {args.per_kinase_csv}")


if __name__ == "__main__":
    main()