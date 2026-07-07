# evaluate_ign.py
#
# Independent test-set evaluation for a trained IGN model.
# Loads a checkpoint .pth file and computes metrics + saves predictions.
#
# Example:
#   python evaluate_ign.py \
#     --test_csv  splits/test.csv \
#     --cache_dir ./ign_cache \
#     --ckpt      ./model_save/2026-04-30_rep0_081523.pth \
#     --gpuid     0

import os
import argparse
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from dataset_ign import IGNDataset, collate_fn_v2_MulPro
from model import DTIPredictorV4_V2


def run_eval(model, dataloader, device):
    """Single-pass inference; returns lists of predictions, truths, keys."""
    true, pred, keys = [], [], []
    model.eval()
    with torch.no_grad():
        for bg, bg3, Ys, ks in dataloader:
            bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
            pred.append(model(bg, bg3).data.cpu().numpy())
            true.append(Ys.data.cpu().numpy())
            keys.append(ks)
    return true, pred, keys


def compute_metrics(true, pred):
    """Standard regression metrics."""
    return {
        'rmse'    : np.sqrt(mean_squared_error(true, pred)),
        'r2'      : r2_score(true, pred),
        'mae'     : mean_absolute_error(true, pred),
        'pearson' : pearsonr(true, pred)[0],
        'spearman': spearmanr(true, pred)[0],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--test_csv',   type=str, required=True,
                        help='CSV for test split (must match training schema)')
    parser.add_argument('--cache_dir',  type=str, default='./ign_cache',
                        help='Root cache dir; expects {cache_dir}/test/ subdir')
    parser.add_argument('--ckpt',       type=str, required=True,
                        help='Path to trained model .pth file')

    # runtime
    parser.add_argument('--gpuid',       type=str, default='0')
    parser.add_argument('--batch_size',  type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_process', type=int, default=8,
                        help='Workers for graph build (only if cache missing)')
    parser.add_argument('--rebuild_cache', action='store_true')

    # model architecture — must match training!
    parser.add_argument('--node_feat_size',  type=int, default=12)
    parser.add_argument('--edge_feat_size',  type=int, default=11)
    parser.add_argument('--graph_feat_size', type=int, default=256)
    parser.add_argument('--num_layers',      type=int, default=3)
    parser.add_argument('--outdim_g3',       type=int, default=200)
    parser.add_argument('--d_FC_layer',      type=int, default=200)
    parser.add_argument('--n_FC_layer',      type=int, default=2)
    parser.add_argument('--dropout',         type=float, default=0.1)
    parser.add_argument('--n_tasks',         type=int,   default=1)

    # output
    parser.add_argument('--out_dir', type=str, default='./eval_results')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # disable cudnn benchmark to avoid Hopper SM_90 kernel selection issues
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True

    # ---- dataset ---------------------------------------------------------
    print('=== Loading test dataset ===')
    test_dataset = IGNDataset(
        csv_path    = args.test_csv,
        cache_dir   = os.path.join(args.cache_dir, 'test'),
        inter_dist  = 5.0,
        num_process = args.num_process,
        rebuild     = args.rebuild_cache,
    )
    print(f'  test samples: {len(test_dataset)}')

    test_loader = DataLoader(
        test_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_v2_MulPro,
    )

    # ---- model -----------------------------------------------------------
    print('=== Loading model ===')
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
    model = DTIPredictorV4_V2(
        node_feat_size = args.node_feat_size,
        edge_feat_size = args.edge_feat_size,
        num_layers     = args.num_layers,
        graph_feat_size= args.graph_feat_size,
        outdim_g3      = args.outdim_g3,
        d_FC_layer     = args.d_FC_layer,
        n_FC_layer     = args.n_FC_layer,
        dropout        = args.dropout,
        n_tasks        = args.n_tasks,
    ).to(device)

    print(f'  loading checkpoint: {args.ckpt}')
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # ---- inference -------------------------------------------------------
    print('=== Running inference ===')
    test_true, test_pred, test_keys = run_eval(model, test_loader, device)

    test_true = np.concatenate(test_true).flatten()
    test_pred = np.concatenate(test_pred).flatten()
    test_keys = [k for batch in test_keys for k in batch]

    # ---- save predictions ------------------------------------------------
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    pred_csv = os.path.join(args.out_dir, f'{ckpt_name}_test_predictions_{ts}.csv')

    pd.DataFrame({
        'key' : test_keys,
        'true': test_true,
        'pred': test_pred,
    }).to_csv(pred_csv, index=False)
    print(f'  predictions saved: {pred_csv}')

    # ---- metrics ---------------------------------------------------------
    m = compute_metrics(test_true, test_pred)

    print('\n=== Test set metrics ===')
    print(f'  RMSE       : {m["rmse"]:.4f}')
    print(f'  MAE        : {m["mae"]:.4f}')
    print(f'  R²         : {m["r2"]:.4f}')
    print(f'  Pearson r  : {m["pearson"]:.4f}')
    print(f'  Spearman ρ : {m["spearman"]:.4f}')

    metrics_csv = os.path.join(args.out_dir, f'{ckpt_name}_test_metrics_{ts}.csv')
    pd.DataFrame([m]).to_csv(metrics_csv, index=False)
    print(f'  metrics saved: {metrics_csv}')