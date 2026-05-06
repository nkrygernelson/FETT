"""
Train the multi-fidelity base bandgap model.

Usage:
    uv run invoke train
    # or directly:
    uv run src/fett/train.py [hydra-overrides]
"""
import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from fett.callbacks import EpochPrint
from fett.data.data import FettDataset
from fett.model.lightning_module import FettLightningModule


def _build_logger(cfg: DictConfig):
    wb = cfg.training.get("logger", {}).get("wandb", {})
    if not wb or not wb.get("enabled", False):
        return True  # Lightning default CSVLogger
    from lightning.pytorch.loggers import WandbLogger
    return WandbLogger(
        project=wb.get("project", "fett"),
        name=wb.get("name"),
        tags=list(wb.get("tags", [])) or None,
    )


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_set = FettDataset(cfg.data.processed_data_dir, split="train", target_col=cfg.data.target_col)
    val_set = FettDataset(cfg.data.processed_data_dir, split="val", target_col=cfg.data.target_col)
    test_set = FettDataset(cfg.data.processed_data_dir, split="test", target_col=cfg.data.target_col)

    train_dl = torch.utils.data.DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_set, batch_size=cfg.training.batch_size, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=cfg.training.batch_size, num_workers=0)

    # Embed model architecture in checkpoint so it can always be reconstructed.
    model_cfg = OmegaConf.to_container(cfg.model.model, resolve=True)
    model_cfg.pop("_target_", None)
    # Build the inner pytorch model explicitly so Hydra does NOT recurse into
    # `training_cfg.optimizer` (which has its own `_target_` but needs `params` at call time).
    inner_model = instantiate(cfg.model.model)
    module = FettLightningModule(model=inner_model, training_cfg=cfg.training, model_cfg=model_cfg)

    callbacks = [
        EpochPrint(),
        EarlyStopping(
            monitor=cfg.training.callbacks.early_stopping.monitor,
            patience=cfg.training.callbacks.early_stopping.patience,
            mode=cfg.training.callbacks.early_stopping.mode,
        ),
        ModelCheckpoint(
            monitor=cfg.training.callbacks.model_checkpoint.monitor,
            mode=cfg.training.callbacks.model_checkpoint.mode,
            save_top_k=cfg.training.callbacks.model_checkpoint.save_top_k,
            filename=cfg.training.callbacks.model_checkpoint.filename,
            dirpath=cfg.training.callbacks.model_checkpoint.dirpath,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        accelerator=cfg.training.trainer.accelerator,
        gradient_clip_val=cfg.training.trainer.gradient_clip_val,
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        enable_progress_bar=cfg.training.trainer.get("enable_progress_bar", False),
        callbacks=callbacks,
        logger=_build_logger(cfg),
    )
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.test(module, dataloaders=test_dl)


if __name__ == "__main__":
    main()
