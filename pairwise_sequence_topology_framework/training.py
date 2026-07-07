import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
import sys, os
import pandas as pd
from models.ginconv_pair import GINConvNet
from models.gcn_pair import GCNNet
from models.gat_pair import GATNet
from models.gat_gcn_pair import GAT_GCN
from utils import *


# ============================================================
# Training / evaluation functions
# ============================================================

def train(model, device, train_loader, optimizer, epoch, loss_fn):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * TRAIN_BATCH_SIZE,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    all_compound_ids = []
    all_kinase_A_ids = []
    all_kinase_B_ids = []
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            all_compound_ids.extend(data.compound_id.cpu().tolist())
            all_kinase_A_ids.extend(data.kinase_A_id)
            all_kinase_B_ids.extend(data.kinase_B_id)
    return (total_labels.numpy().flatten(),
            total_preds.numpy().flatten(),
            all_compound_ids,
            all_kinase_A_ids,
            all_kinase_B_ids)


def set_seed(seed: int):
    """Seed all RNG sources for reproducibility within a rep."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_one_rep(rep_id: int, seed: int, modeling, pool_method, device,
                train_data, val_data, test_data, model_st):
    """Run one full training repetition and return its test metrics + preds."""
    print(f"\n{'=' * 70}")
    print(f"REP {rep_id}  (seed={seed})  model={model_st}  pool={pool_method}")
    print(f"{'=' * 70}\n")

    set_seed(seed)

    # ---- Loaders (rebuild per rep so shuffle uses fresh RNG state) ----
    # Use a generator so train shuffling is reproducible given the seed.
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True, generator=g)
    val_loader   = DataLoader(val_data,   batch_size=TEST_BATCH_SIZE,
                              shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=TEST_BATCH_SIZE,
                              shuffle=False)

    # ---- Fresh model per rep ----
    model = modeling(pool_method=pool_method).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mse = float('inf')
    best_epoch = -1
    patience_counter = 0

    model_file_name = f'model_{model_st}_{pool_method}_rep{rep_id}.model'
    print(f'Save to: {model_file_name}')

    # ---- Train ----
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1, loss_fn)

        G_val, P_val, _, _, _ = predicting(model, device, val_loader)
        val_ret = [rmse(G_val, P_val), mse(G_val, P_val),
                   pearson(G_val, P_val), spearman(G_val, P_val)]

        print(f'[rep {rep_id}] Val epoch {epoch + 1}: RMSE={val_ret[0]:.4f}, '
              f'MSE={val_ret[1]:.4f}, Pearson={val_ret[2]:.4f}, '
              f'Spearman={val_ret[3]:.4f}')

        if val_ret[1] < best_val_mse:
            best_val_mse = val_ret[1]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), model_file_name)
            print(f'  ** [rep {rep_id}] Val MSE improved at epoch {best_epoch}: '
                  f'{best_val_mse:.4f}')
        else:
            patience_counter += 1
            print(f'  [rep {rep_id}] No improvement since epoch {best_epoch}, '
                  f'best val MSE: {best_val_mse:.4f}, '
                  f'patience: {patience_counter}/{PATIENCE}')
            if patience_counter >= PATIENCE:
                print(f'\n[rep {rep_id}] Early stopping at epoch {epoch + 1}')
                break

    # ---- Final test with best checkpoint ----
    print(f'\n[rep {rep_id}] Loading best model from epoch {best_epoch}...')
    model.load_state_dict(torch.load(model_file_name, weights_only=False))
    G_test, P_test, compound_ids, kinase_A_ids, kinase_B_ids = \
        predicting(model, device, test_loader)
    test_ret = [rmse(G_test, P_test), mse(G_test, P_test),
                pearson(G_test, P_test), spearman(G_test, P_test)]

    print(f'\n===== [rep {rep_id}] Test Results ({model_st} + {pool_method}) =====')
    print(f'RMSE:     {test_ret[0]:.4f}')
    print(f'MSE:      {test_ret[1]:.4f}')
    print(f'Pearson:  {test_ret[2]:.4f}')
    print(f'Spearman: {test_ret[3]:.4f}')
    print(f'Best epoch: {best_epoch}')

    # ---- Save per-rep predictions ----
    pred_file_name = f'pred_{model_st}_{pool_method}_rep{rep_id}.csv'
    pred_df = pd.DataFrame({
        'compound_id': compound_ids,
        'kinase_A':    kinase_A_ids,
        'kinase_B':    kinase_B_ids,
        'pred_score':  P_test,
        'true_label':  G_test,
    })
    pred_df.to_csv(pred_file_name, index=False)
    print(f'[rep {rep_id}] Predictions saved to {pred_file_name} '
          f'({len(pred_df)} samples)')

    return {
        'rep':       rep_id,
        'seed':      seed,
        'best_epoch': best_epoch,
        'rmse':      test_ret[0],
        'mse':       test_ret[1],
        'pearson':   test_ret[2],
        'spearman':  test_ret[3],
        'pred_df':   pred_df,
    }


# ============================================================
# Main
# ============================================================
# Usage:
#   python training.py <model_id> <pool_method> [cuda_id] [n_reps]
#
#   model_id:   0=GINConv, 1=GCN, 2=GAT, 3=GAT_GCN
#   pool_method: diff, diff_lig, concat_just, concat_full
#   cuda_id:    GPU index (default 0)
#   n_reps:     number of repetitions (default 3)
#
# Example: python training.py 0 diff_lig 0 3
# ============================================================

modeling    = [GINConvNet, GCNNet, GATNet, GAT_GCN][int(sys.argv[1])]
pool_method = sys.argv[2] if len(sys.argv) > 2 else 'diff'
cuda_name   = "cuda:" + str(int(sys.argv[3])) if len(sys.argv) > 3 else "cuda:0"
N_REPS      = int(sys.argv[4]) if len(sys.argv) > 4 else 3

# Fixed seeds — chosen for consistency with your other multi-seed work
SEEDS = [42]
if N_REPS > len(SEEDS):
    # Extend deterministically if more reps requested
    SEEDS = SEEDS + list(range(2025, 2025 + (N_REPS - len(SEEDS))))
SEEDS = SEEDS[:N_REPS]

model_st = modeling.__name__
print(f'Model: {model_st}, Pool: {pool_method}, Device: {cuda_name}')
print(f'N_REPS: {N_REPS}, Seeds: {SEEDS}')

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE  = 256
LR               = 0.0005
LOG_INTERVAL     = 20
NUM_EPOCHS       = 100
PATIENCE         = 15

print(f'Learning rate: {LR}, Epochs: {NUM_EPOCHS}, Patience: {PATIENCE}')

# ---- Load data ONCE — graphs are identical across reps ----
print('\nLoading datasets...')
train_data = SiameseTestbedDataset(root='data_random_split', dataset='selectivity_train_pocket_gta')
val_data   = SiameseTestbedDataset(root='data_random_split', dataset='selectivity_valid_pocket_gta')
test_data  = SiameseTestbedDataset(root='data_random_split', dataset='selectivity_test_pocket_gta')
print(f'  train: {len(train_data)}  val: {len(val_data)}  test: {len(test_data)}')

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

# ---- Run all reps ----
all_results = []
for rep_id, seed in enumerate(SEEDS):
    result = run_one_rep(
        rep_id=rep_id, seed=seed,
        modeling=modeling, pool_method=pool_method, device=device,
        train_data=train_data, val_data=val_data, test_data=test_data,
        model_st=model_st,
    )
    all_results.append(result)

# ============================================================
# Aggregate across reps
# ============================================================
print(f"\n{'=' * 70}")
print(f"CROSS-REP SUMMARY  ({model_st} + {pool_method}, n={N_REPS})")
print(f"{'=' * 70}")

per_rep_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'pred_df'}
    for r in all_results
])
print("\nPer-rep test metrics:")
print(per_rep_df.to_string(index=False))

# Mean ± std across reps
metric_cols = ['rmse', 'mse', 'pearson', 'spearman']
agg = per_rep_df[metric_cols].agg(['mean', 'std'])
print("\nAggregate (mean ± std):")
for m in metric_cols:
    mean = agg.loc['mean', m]
    std  = agg.loc['std', m]
    print(f'  {m:>9s}: {mean:.4f} ± {std:.4f}')

# Save per-rep metrics CSV
result_file_name = f'result_{model_st}_{pool_method}_allreps.csv'
per_rep_df.to_csv(result_file_name, index=False)
print(f'\nPer-rep metrics saved to {result_file_name}')

# Save aggregate summary CSV
agg_file_name = f'result_{model_st}_{pool_method}_summary.csv'
summary_rows = []
for m in metric_cols:
    summary_rows.append({
        'metric': m,
        'mean':   agg.loc['mean', m],
        'std':    agg.loc['std', m],
    })
pd.DataFrame(summary_rows).to_csv(agg_file_name, index=False)
print(f'Aggregate summary saved to {agg_file_name}')

# ---- Optional: ensemble predictions across reps ----
# Average predictions over reps, keyed on (compound_id, kinase_A, kinase_B).
# Useful if you want a single ensembled prediction set for downstream analysis.
print("\nBuilding ensemble predictions across reps...")
first = all_results[0]['pred_df'].copy()
first = first.rename(columns={'pred_score': 'pred_rep0'})
for i, r in enumerate(all_results[1:], start=1):
    col = f'pred_rep{i}'
    first[col] = r['pred_df']['pred_score'].values

pred_cols = [f'pred_rep{i}' for i in range(N_REPS)]
first['pred_mean'] = first[pred_cols].mean(axis=1)
first['pred_std']  = first[pred_cols].std(axis=1)

ensemble_file = f'pred_{model_st}_{pool_method}_ensemble.csv'
first.to_csv(ensemble_file, index=False)
print(f'Ensemble predictions saved to {ensemble_file}')

# Ensemble metrics
ens_rmse     = rmse(first['true_label'].values, first['pred_mean'].values)
ens_mse      = mse(first['true_label'].values, first['pred_mean'].values)
ens_pearson  = pearson(first['true_label'].values, first['pred_mean'].values)
ens_spearman = spearman(first['true_label'].values, first['pred_mean'].values)
print(f'\nEnsemble (mean of {N_REPS} reps) test metrics:')
print(f'  RMSE:     {ens_rmse:.4f}')
print(f'  MSE:      {ens_mse:.4f}')
print(f'  Pearson:  {ens_pearson:.4f}')
print(f'  Spearman: {ens_spearman:.4f}')