"""
Train KDBNet's DTAModel on cached graphs from your parquet pipeline.

This is a self-contained training script that uses:
  - kdbnet.model.DTAModel       (architecture, unchanged from KDBNet)
  - kdbnet_dataset.KDBNetCachedDataset (your cached PyG graphs)

It does NOT use KDBNet's DTAExperiment, DTATask, or DTA classes -- those
are tightly coupled to KDBNet's PDB/JSON data layout and would require
extensive monkey-patching to work with your parquet-derived caches.

Usage
-----
    cd /home/qinli_cluster/prediction_model/selectivity/KDBNet
    export PYTHONPATH=$PWD:$PYTHONPATH

    python scripts/train.py \
        --train_csv data/train.csv \
        --valid_csv data/valid.csv \
        --test_csv data/test.csv \
        --prot_dir data/kdbnet/proteins \
        --drug_dir data/kdbnet/drugs \
        --out_dir outputs/run1 \
        --batch_size 64 \
        --lr 5e-4 \
        --n_epochs 100 \
        --patience 15 \
        --device cuda
"""
import argparse
import json
import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from kdbnet.model import DTAModel
from scripts.kdbnet_dataset import KDBNetCachedDataset, kdbnet_collate
# If kdbnet_dataset.py lives somewhere else, adjust the import accordingly.


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    pearson = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    spearman = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "pearson": pearson, "spearman": spearman}


def per_group_metrics(df, group_col, min_n=5):
    """Compute per-group (per-kinase or per-compound) metrics.

    Groups with fewer than min_n samples get NaN for correlation metrics
    (correlations on tiny groups are noisy).
    """
    rows = []
    for grp_val, sub in df.groupby(group_col):
        n = len(sub)
        m = compute_metrics(sub["y_true"].values, sub["y_pred"].values) if n >= min_n \
            else {"mse": float(np.mean((sub["y_true"] - sub["y_pred"]) ** 2)),
                  "rmse": float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2))),
                  "mae": float(np.mean(np.abs(sub["y_true"] - sub["y_pred"]))),
                  "pearson": float("nan"), "spearman": float("nan")}
        rows.append({group_col: grp_val, "n": n, **m})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        drug = batch["drug"].to(device)
        prot = batch["protein"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        pred = model(drug, prot).squeeze(-1)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, return_predictions=False):
    model.eval()
    ys, preds = [], []
    drug_names, prot_names = [], []
    for batch in loader:
        drug = batch["drug"].to(device)
        prot = batch["protein"].to(device)
        y = batch["y"]
        pred = model(drug, prot).squeeze(-1).cpu()
        ys.append(y)
        preds.append(pred)
        # If you want per-pair output, the dataset returns names; PyG's
        # Batch carries `.name` lists for each graph.
        drug_names.extend(batch["drug"].name)
        prot_names.extend(batch["protein"].name)
    if not ys:
        raise RuntimeError(
            "evaluate() got an empty loader. Likely the dataset dropped all rows "
            "because cached .pt files for the test/valid kinases or compounds are "
            "missing. Check that steps 02 and 04 were run on a CSV that includes "
            "ALL splits (train + valid + test concatenated)."
        )
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    metrics = compute_metrics(y_true, y_pred)
    if return_predictions:
        df = pd.DataFrame({
            "compound_id": drug_names,
            "kinase": prot_names,
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_pred - y_true,
            "abs_error": np.abs(y_pred - y_true),
        })
        return metrics, df
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--valid_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--prot_dir", default="data/kdbnet/proteins")
    parser.add_argument("--drug_dir", default="data/kdbnet/drugs")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--preload", action="store_true",
                        help="Load all graphs into RAM up front (faster training, "
                             "needs ~5-10 GB for typical kinase datasets).")

    # Model hyperparameters (defaults match KDBNet's DTAModel defaults)
    parser.add_argument("--prot_emb_dim", type=int, default=1280)
    parser.add_argument("--prot_gcn_dims", type=int, nargs="+", default=[128, 256, 256])
    parser.add_argument("--prot_fc_dims", type=int, nargs="+", default=[1024, 128])
    parser.add_argument("--drug_node_h_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--drug_edge_h_dims", type=int, nargs="+", default=[32, 1])
    parser.add_argument("--drug_fc_dims", type=int, nargs="+", default=[1024, 128])
    parser.add_argument("--mlp_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--mlp_dropout", type=float, default=0.25)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience on val pearson")
    parser.add_argument("--monitor", default="pearson",
                        choices=["pearson", "spearman", "rmse", "mse", "mae"])
    parser.add_argument("--num_workers", type=int, default=4)

    # I/O
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_predictions", action="store_true", default=True,
                        help="Save per-pair predictions to CSV (default: True). "
                             "Pass --no_save_predictions to disable.")
    parser.add_argument("--no_save_predictions", action="store_false",
                        dest="save_predictions")
    parser.add_argument("--save_train_predictions", action="store_true",
                        help="Also save predictions on the training set (slow). "
                             "Useful for diagnosing overfitting.")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint .pt to resume from. "
                             "If given, model + optimizer + epoch + best score "
                             "are restored, and training continues for "
                             "--n_epochs MORE epochs (not 'until' that epoch).")

    args = parser.parse_args()

    # ---- Setup ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    # ---- Data ----
    print(f"[data] loading splits...")
    train_ds = KDBNetCachedDataset(args.train_csv, args.prot_dir, args.drug_dir,
                                    label_col=args.label_col, preload=args.preload)
    valid_ds = KDBNetCachedDataset(args.valid_csv, args.prot_dir, args.drug_dir,
                                    label_col=args.label_col, preload=args.preload)
    test_ds  = KDBNetCachedDataset(args.test_csv,  args.prot_dir, args.drug_dir,
                                    label_col=args.label_col, preload=args.preload)
    print(f"[data] train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

    # PyG batches are slow to pickle for multiprocessing; if num_workers>0
    # causes hangs, drop to 0.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=kdbnet_collate, num_workers=args.num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=kdbnet_collate, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=kdbnet_collate, num_workers=args.num_workers,
                              pin_memory=True)

    # ---- Model ----
    model = DTAModel(
        prot_emb_dim=args.prot_emb_dim,
        prot_gcn_dims=args.prot_gcn_dims,
        prot_fc_dims=args.prot_fc_dims,
        drug_node_h_dims=args.drug_node_h_dims,
        drug_edge_h_dims=args.drug_edge_h_dims,
        drug_fc_dims=args.drug_fc_dims,
        mlp_dims=args.mlp_dims,
        mlp_dropout=args.mlp_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DTAModel, trainable params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # ---- Training ----
    best_score = -np.inf if args.monitor in ("pearson", "spearman") else np.inf
    best_epoch = -1
    bad_epochs = 0
    history = []
    start_epoch = 1

    # Resume
    if args.resume:
        print(f"[resume] loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print("[resume] no optimizer state in checkpoint; using fresh Adam")
        best_score = ckpt.get("best_score", best_score)
        best_epoch = ckpt.get("best_epoch", best_epoch)
        start_epoch = ckpt.get("epoch", 0) + 1
        # Optionally restore prior history
        prior_history_path = Path(args.resume).parent / "history.csv"
        if prior_history_path.exists():
            history = pd.read_csv(prior_history_path).to_dict("records")
            print(f"[resume] loaded {len(history)} prior epochs from {prior_history_path}")
        print(f"[resume] continuing from epoch {start_epoch}, "
              f"best so far: epoch {best_epoch}, val {args.monitor}={best_score:.4f}")

    def is_better(new, old):
        if args.monitor in ("pearson", "spearman"):
            return new > old
        return new < old

    end_epoch = start_epoch + args.n_epochs - 1 if args.resume else args.n_epochs
    print(f"[train] starting (epochs {start_epoch}..{end_epoch}), "
          f"monitoring val {args.monitor}, patience={args.patience}")
    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, valid_loader, device)
        elapsed = time() - t0

        score = val_metrics[args.monitor]
        log_row = {"epoch": epoch, "train_loss": train_loss, "elapsed": elapsed}
        log_row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(log_row)
        print(f"[ep {epoch:3d}] loss={train_loss:.4f}  "
              f"val pearson={val_metrics['pearson']:.4f}  "
              f"rmse={val_metrics['rmse']:.4f}  "
              f"({elapsed:.1f}s)")

        # Always save the latest checkpoint (for resuming)
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "args": vars(args),
            # Just the model-architecture subset of args, for easy inference loading
            "model_kwargs": {
                "prot_emb_dim": args.prot_emb_dim,
                "prot_gcn_dims": args.prot_gcn_dims,
                "prot_fc_dims": args.prot_fc_dims,
                "drug_node_h_dims": args.drug_node_h_dims,
                "drug_edge_h_dims": args.drug_edge_h_dims,
                "drug_fc_dims": args.drug_fc_dims,
                "mlp_dims": args.mlp_dims,
                "mlp_dropout": args.mlp_dropout,
            },
        }
        torch.save(ckpt, out_dir / "last_checkpoint.pt")

        if is_better(score, best_score):
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            # Save best checkpoint with weights + hyperparameters bundled
            best_ckpt = {
                "model": model.state_dict(),
                "model_kwargs": ckpt["model_kwargs"],
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            torch.save(best_ckpt, out_dir / "best_model.pt")
            # Update best info inside last_checkpoint too
            ckpt["best_score"] = best_score
            ckpt["best_epoch"] = best_epoch
            torch.save(ckpt, out_dir / "last_checkpoint.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, val {args.monitor}={best_score:.4f})")
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    print(f"[train] best epoch {best_epoch}, val {args.monitor}={best_score:.4f}")

    # ---- Test ----
    print("[test] loading best checkpoint and evaluating on test set")
    best = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    # New format: dict with 'model' key. Old format: plain state_dict.
    if isinstance(best, dict) and "model" in best:
        model.load_state_dict(best["model"])
    else:
        model.load_state_dict(best)
    test_metrics, test_df = evaluate(model, test_loader, device, return_predictions=True)
    print(f"[test] " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Always save predictions (toggleable via --no_save_predictions)
    if args.save_predictions:
        test_df.to_csv(out_dir / "test_predictions.csv", index=False)
        print(f"[test] saved {len(test_df)} predictions to test_predictions.csv")

        # Per-kinase metrics: how does the model perform on each kinase?
        per_kinase = per_group_metrics(test_df, "kinase", min_n=5)
        per_kinase.to_csv(out_dir / "test_metrics_per_kinase.csv", index=False)
        print(f"[test] per-kinase metrics: "
              f"{len(per_kinase)} kinases, "
              f"median pearson={per_kinase['pearson'].median():.4f}, "
              f"top5={per_kinase.nlargest(5, 'pearson')[['kinase','n','pearson']].values.tolist()[:3]}")

        # Per-compound metrics: less interpretable but useful for outlier detection
        per_compound = per_group_metrics(test_df, "compound_id", min_n=5)
        per_compound.to_csv(out_dir / "test_metrics_per_compound.csv", index=False)

    # Validation set predictions (also useful for ensembling / calibration later)
    valid_metrics, valid_df = evaluate(model, valid_loader, device, return_predictions=True)
    with open(out_dir / "valid_metrics.json", "w") as f:
        json.dump(valid_metrics, f, indent=2)
    if args.save_predictions:
        valid_df.to_csv(out_dir / "valid_predictions.csv", index=False)
        per_kinase_v = per_group_metrics(valid_df, "kinase", min_n=5)
        per_kinase_v.to_csv(out_dir / "valid_metrics_per_kinase.csv", index=False)

    # Optionally also predict on the train set (slower; useful for overfitting diagnosis)
    if args.save_train_predictions:
        print("[train-eval] computing predictions on full training set "
              "(slow; pass --no_save_train_predictions to skip)")
        train_metrics, train_df = evaluate(model, train_loader, device,
                                            return_predictions=True)
        train_df.to_csv(out_dir / "train_predictions.csv", index=False)
        with open(out_dir / "train_metrics.json", "w") as f:
            json.dump(train_metrics, f, indent=2)
        print(f"[train-eval] " + "  ".join(f"{k}={v:.4f}"
                                            for k, v in train_metrics.items()))

    print(f"[done] artifacts in {out_dir}")


if __name__ == "__main__":
    main()