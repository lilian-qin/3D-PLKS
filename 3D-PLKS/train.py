import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

import yaml
from collections import defaultdict
from pathlib import Path
import argparse
import shutil

from models.egnn_model import ddgEGNN

from models.pic50_model import pIC50EGNN
from base.dataset import ddgDataSet, PLAffinityDataSet
from pytorch_lightning.callbacks import EarlyStopping

def main(config: dict):
    """Build EGNN model using params specified in config file"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["model"] == "ddgEGNN":
        ModelClass = ddgEGNN
    else:
        ModelClass = pIC50EGNN
        #raise NotImplementedError

    # Set up model with parameters specified in config file
    model = ModelClass(
        dataset_config=config["dataset_params"],
        loader_config=config["loader_params"],
        trainer_config=config["trainer_params"],
        **config["model_params"]
    )

    # Options to load model checkpoints
    ckpt_path = None
    if config["restore"] is not None:
        ckpt_path = config["restore"]
        print(f"Will resume training from checkpoint: {ckpt_path}")

    if config["initialize_weights"] is not None:
        checkpoint = torch.load(config["initialize_weights"]["checkpoint_file"], map_location=device)
        pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items()}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded model weights from {config['initialize_weights']['checkpoint_file']}")

    # Define trainer
    trainer_config = config["trainer_params"].copy() if config["trainer_params"] else {}

    # Remove deprecated parameters
    deprecated_params = ["resume_from_checkpoint", "gpus", "auto_select_gpus"]
    for param in deprecated_params:
        if param in trainer_config:
            del trainer_config[param]

    # Handle GPU configuration for PL 2.x
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
    if "devices" not in trainer_config:
        trainer_config["devices"] = 1

    if config["logger_params"]["wandb_bool"]:
        logger = WandbLogger(
            save_dir=Path(config["save_dir"]),
            offline=False,
            project=f"{config['logger_params']['wandb']}",
            name=f"{config['name']}",
            group=config["logger_params"]["group"],
        )
        logger.log_hyperparams({
            "graph_generation_mode": config["dataset_params"]["graph_generation_mode"],
            **config["model_params"],
        })
    else:
        logger = TensorBoardLogger(save_dir=config["save_dir"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        filename="{epoch}-{val_mae:.4f}",
        dirpath=config["save_dir"],
    )

    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        patience=5,
                                        mode="min",
                                        min_delta=0.001,)

    # Set up model training
    trainer = pl.Trainer(
        default_root_dir=config["save_dir"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        **trainer_config
    )

    # Train model
    if config["train"]:
        trainer.fit(model, ckpt_path=ckpt_path)

    # Test model
    if config["test"]:
        model.test_set_predictions = []
        trainer.test(model, ckpt_path="best")
        model.save_test_predictions(Path(config["save_dir"]) / f"preds_{config['name']}.csv")

    # Save final model parameters
    torch.save({
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "model_state_dict": model.state_dict(),
        "best_val_pearson": checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        "best_model_path": checkpoint_callback.best_model_path,
    }, config["save_dir"] + f"checkpoint_final_{config['name']}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config) as yaml_file_handle:
        config = yaml.safe_load(yaml_file_handle)
    config = defaultdict(lambda: None, config)

    config["save_dir"] = config["save_dir"] + config["name"] + "/"
    Path(config["save_dir"]).mkdir(exist_ok=True, parents=True)

    shutil.copyfile(args.config, config["save_dir"] + f"config_{config['name']}.yaml")

    main(config)