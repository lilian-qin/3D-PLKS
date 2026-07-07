# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import wandb
from utils import AverageMeter, BestMeter, save_model_dict, load_model_dict
from GIGN import GIGN
from dataset import SingleComplexDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
from sklearn.metrics import mean_squared_error


# %%
def val(model, dataloader, device):
    model.eval()
    pred_list = []
    label_list = []
    compound_ids = []
    kinases = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(data.y.detach().cpu().numpy())
            compound_ids.extend(data.compound_id)
            kinases.extend(data.kinase)

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()
    return rmse, coff, pred, label, compound_ids, kinases


# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN'
    config = Config(cfg)
    args = config.get_config()
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")

    for repeat in range(repeats):
        args['repeat'] = repeat

        train_set = SingleComplexDataset(os.path.join(data_root, 'train_pic50.csv'))
        valid_set = SingleComplexDataset(os.path.join(data_root, 'val_pic50.csv'))
        test_set = SingleComplexDataset(os.path.join(data_root, 'test_pic50.csv'))

        train_loader = PLIDataLoader(train_set, batch_size=batch_size,
                                     shuffle=True, num_workers=8,
                                     persistent_workers=True, pin_memory=True)
        valid_loader = PLIDataLoader(valid_set, batch_size=batch_size,
                                     shuffle=False, num_workers=4,
                                     persistent_workers=True, pin_memory=True)
        test_loader = PLIDataLoader(test_set, batch_size=batch_size,
                                    shuffle=False, num_workers=4,
                                    persistent_workers=True, pin_memory=True)

        logger = TrainLogger(args, cfg, create=True)
        logger.info(f"train: {len(train_set)}, valid: {len(valid_set)}, test: {len(test_set)}")

        device = torch.device('cuda:0')
        model = GIGN(13, 256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []

        model.train()
        for epoch in range(epochs):
            for data in train_loader:
                data = data.to(device)
                pred = model(data)
                label = data.y

                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0))

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            valid_rmse, valid_pr, _, _, _, _ = val(model, valid_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info("best_rmse: %.4f" % running_best_mse.get_best())
                    break

        # --- Final evaluation with best model ---
        # --- Final evaluation ---
        load_model_dict(model, best_model_list[-1])

        valid_rmse, valid_pr, _, _, _, _ = val(model, valid_loader, device)
        test_rmse, test_pr, preds, labels, compound_ids, kinases = val(model, test_loader, device)

        msg = "valid_rmse-%.4f, valid_pr-%.4f, test_rmse-%.4f, test_pr-%.4f" \
                    % (valid_rmse, valid_pr, test_rmse, test_pr)
        logger.info(msg)

        # Save test predictions
        test_df = pd.DataFrame({
            'compound_id': compound_ids,
            'kinase': kinases,
            'true_label': labels,
            'pred_score': preds,
        })
        pred_save_path = os.path.join(logger.get_model_dir(), 'test_predictions.csv')
        test_df.to_csv(pred_save_path, index=False)
        logger.info(f"Test predictions saved to {pred_save_path}")
        