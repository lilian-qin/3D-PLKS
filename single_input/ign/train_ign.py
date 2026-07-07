# ign_train.py  —  adapted for kinase parquet pipeline
import datetime
import time
import os
import warnings
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

from dataset_ign import IGNDataset, collate_fn_v2_MulPro
from utils import set_random_seed, EarlyStopping
from model import DTIPredictorV4_V2

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

DataLoaderX = DataLoader


# ---------------------------------------------------------------------------
# Train / eval loops  (unchanged from original IGN)
# ---------------------------------------------------------------------------



def run_a_train_epoch(model, loss_fn, dataloader, optimizer, device):
    model.train()
    for bg, bg3, Ys, _ in dataloader:
        model.zero_grad()
        bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
        loss = loss_fn(model(bg, bg3), Ys)
        loss.backward()
        optimizer.step()
'''
def run_a_train_epoch(model, loss_fn, dataloader, optimizer, device):
    model.train()
    for i_batch, (bg, bg3, Ys, keys) in enumerate(dataloader):
        if i_batch == 0:
            print(f'BATCH 0 ACTUAL SIZE: {len(keys)}')
        try:
            model.zero_grad()
            bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
            loss = loss_fn(model(bg, bg3), Ys)
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            print(f'\n[CRASH] batch {i_batch}, keys in batch:')
            for k in keys:
                print(f'  {k}')
            raise
'''
def run_a_eval_epoch(model, dataloader, device):
    true, pred, keys = [], [], []
    model.eval()
    with torch.no_grad():
        for bg, bg3, Ys, ks in dataloader:
            bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
            pred.append(model(bg, bg3).data.cpu().numpy())
            true.append(Ys.data.cpu().numpy())
            keys.append(ks)
    return true, pred, keys

def _metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rp = pearsonr(y_true, y_pred)[0]

    return rmse, r2, mae, rp
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ---- data paths -------------------------------------------------------
    parser.add_argument('--train_csv',  type=str, default='../data_process/dta_data/train_pic50.csv',
                        help='CSV for training split')
    parser.add_argument('--valid_csv',  type=str, default='../data_process/dta_data/val_pic50.csv',
                        help='CSV for validation split')
    parser.add_argument('--test_csv',   type=str, default='../data_process/dta_data/test_pic50.csv',
                        help='CSV for test split')
    parser.add_argument('--cache_dir',  type=str, default='./ign_cache',
                        help='Root dir for graph cache (sub-dirs train/valid/test created automatically)')

    # ---- training hyperparams --------------------------------------------
    parser.add_argument('--gpuid',      type=str,   default='0')
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=200)
    parser.add_argument('--tolerance',  type=float, default=0.0)
    parser.add_argument('--patience',   type=int,   default=5)
    parser.add_argument('--l2',         type=float, default=1e-5)
    parser.add_argument('--repetitions',type=int,   default=3)
    parser.add_argument('--rep_id',     type=int,   required=True)
    parser.add_argument('--num_workers',type=int,   default=0)
    parser.add_argument('--num_process',type=int,   default=16,
                        help='Worker processes for graph generation')

    # ---- model hyperparams -----------------------------------------------
    # Defaults updated to match the parquet-based graph schema:
    #   node_feat_size   = max(type_map)+1+1  (type one-hot + formal_charge)
    #   edge_feat_size_2d = 11  ([is_inter|dist|3D-geom])  — intra edges on g
    #   edge_feat_size_3d = 1   ([dist])                   — inter edges on g3
    parser.add_argument('--node_feat_size',    type=int, default=12)#shouldbe12
    parser.add_argument('--edge_feat_size_2d', type=int, default=11)
    parser.add_argument('--edge_feat_size_3d', type=int, default=1)
    parser.add_argument('--edge_feat_size', type=int, default=11,
                    help='Intra-molecular edge feature dim for g '
                         '(is_inter=1 + dist=1 + 3D-geom=9 = 11)')
    parser.add_argument('--graph_feat_size',   type=int, default=200)
    parser.add_argument('--num_layers',        type=int, default=3)
    parser.add_argument('--outdim_g3',         type=int, default=200)
    parser.add_argument('--d_FC_layer',        type=int, default=200)
    parser.add_argument('--n_FC_layer',        type=int, default=2)
    parser.add_argument('--dropout',           type=float, default=0.1)
    parser.add_argument('--n_tasks',           type=int,   default=1)

    # ---- misc ------------------------------------------------------------
    parser.add_argument('--model_save_dir', type=str, default='./model_save')
    parser.add_argument('--rebuild_cache',  action='store_true',
                        help='Force rebuild of graph cache from parquets')

    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs('./stats', exist_ok=True)

    # ---- build datasets (graph generation happens here, once) ------------
    print('=== Preparing datasets ===')
    train_dataset = IGNDataset(
        csv_path    = args.train_csv,
        cache_dir   = os.path.join(args.cache_dir, 'train'),
        inter_dist  = 5.0,
        num_process = args.num_process,
        rebuild     = args.rebuild_cache,
    )
    valid_dataset = IGNDataset(
        csv_path    = args.valid_csv,
        cache_dir   = os.path.join(args.cache_dir, 'valid'),
        inter_dist  = 5.0,
        num_process = args.num_process,
        rebuild     = args.rebuild_cache,
    )
    test_dataset = IGNDataset(
        csv_path    = args.test_csv,
        cache_dir   = os.path.join(args.cache_dir, 'test'),
        inter_dist  = 5.0,
        num_process = args.num_process,
        rebuild     = args.rebuild_cache,
    )

    print(f'  train : {len(train_dataset)}')
    print(f'  valid : {len(valid_dataset)}')
    print(f'  test  : {len(test_dataset)}')

    # ---- repetitions ------------------------------------------------------
    stat_res = []

    #for rep in range(args.repetitions):
        # set_random_seed(rep)   # ← comment out temporarily
    rep = args.rep_id
        
    # dataloaders — test loader built after best model is loaded
    train_loader = DataLoaderX(
        train_dataset, args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn_v2_MulPro,
    )
    valid_loader = DataLoaderX(
        valid_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_v2_MulPro,
    )

    # ---- model --------------------------------------------------------
    
    model = DTIPredictorV4_V2(
        node_feat_size = args.node_feat_size,
        edge_feat_size = args.edge_feat_size,   # g3 inter edges
        num_layers     = args.num_layers,
        graph_feat_size= args.graph_feat_size,
        outdim_g3      = args.outdim_g3,
        d_FC_layer     = args.d_FC_layer,
        n_FC_layer     = args.n_FC_layer,
        dropout        = args.dropout,
        n_tasks        = args.n_tasks,
    )
    if rep == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  parameters: {n_params:,}')
        print(model)

    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    loss_fn   = nn.MSELoss()

    dt       = datetime.datetime.now()
    ckpt_path = os.path.join(
        args.model_save_dir,
        '{}_rep{}_{:02d}{:02d}{:02d}.pth'.format(dt.date(), rep, dt.hour, dt.minute, dt.second),
    )
    stopper = EarlyStopping(mode='lower', patience=args.patience,
                            tolerance=args.tolerance, filename=ckpt_path)

    # ---- training loop ------------------------------------------------
    valid_rmse = float('inf')   # sentinel before first eval
    for epoch in range(args.epochs):
        t0 = time.time()
        run_a_train_epoch(model, loss_fn, train_loader, optimizer, device)

        # Eval every 5 epochs (and on the first epoch)
        if epoch % 5 == 0:
            valid_true, valid_pred, _ = run_a_eval_epoch(model, valid_loader, device)
            valid_true = np.concatenate(valid_true).flatten()
            valid_pred = np.concatenate(valid_pred).flatten()
            valid_rmse = np.sqrt(mean_squared_error(valid_true, valid_pred))

            if stopper.step(valid_rmse, model):
                print(f'  Early stopping at epoch {epoch}')
                break

            print(f'epoch:{epoch:4d}  valid_rmse:{valid_rmse:.4f}  '
                    f'time:{time.time()-t0:.1f}s')
        else:
            print(f'epoch:{epoch:4d}  (skip eval)  '
                    f'time:{time.time()-t0:.1f}s')

    # ---- evaluation on best checkpoint --------------------------------
    # load best checkpoint
    stopper.load_checkpoint(model)

    test_loader = DataLoaderX(
        test_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_v2_MulPro,
    )

    test_true, test_pred, test_keys = run_a_eval_epoch(model, test_loader, device)
    test_true  = np.concatenate(test_true).flatten()
    test_pred  = np.concatenate(test_pred).flatten()
    test_keys  = [k for batch in test_keys for k in batch]

    # save test predictions
    ts = '{}_rep{}_{:02d}{:02d}{:02d}'.format(dt.date(), rep, dt.hour, dt.minute, dt.second)
    pd.DataFrame({'key': test_keys, 'true': test_true, 'pred': test_pred})\
    .to_csv(f'./stats/{ts}_te.csv', index=False)

    te_rmse, te_r2, te_mae, te_rp = _metrics(test_true, test_pred)
    print(f'test   rmse:{te_rmse:.4f}  r2:{te_r2:.4f}  mae:{te_mae:.4f}  rp:{te_rp:.4f}')

    stat_res.extend([
        [rep, 'test', te_rmse, te_r2, te_mae, te_rp],
    ])

    # ---- aggregate across repetitions ------------------------------------
    stat_df = pd.DataFrame(stat_res, columns=['rep', 'group', 'rmse', 'r2', 'mae', 'rp'])
    stat_df.to_csv(f'./stats/{ts}_all.csv', index=False)

    print(f"rep {rep} done. test rmse={te_rmse:.4f}, r2={te_r2:.4f}, "
      f"mae={te_mae:.4f}, rp={te_rp:.4f}")