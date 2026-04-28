"""
Train the FidelityTranslationHead on top of a frozen pre-trained base model.

The base model checkpoint must have been saved by FettLightningModule (with embedded model_cfg).
The translation dataset (matched fidelity pairs) must already be generated:
    uv run invoke make-data-translation

Example:
    uv run invoke train-translation --base-ckpt models/best.ckpt
    # or directly:
    uv run src/fett/train_translation.py model.base_model_checkpoint=models/best.ckpt
"""
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch

from fett.data.data import FettTranslationDataset
from fett.model.lightning_module import FettTranslationModule


@hydra.main(config_path="../../configs", config_name="config_translation", version_base="1.3")
def main(cfg: DictConfig):
    train_set = FettTranslationDataset(cfg.data.processed_data_dir, split="train")
    val_set = FettTranslationDataset(cfg.data.processed_data_dir, split="val")
    test_set = FettTranslationDataset(cfg.data.processed_data_dir, split="test")

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.training.batch_size, num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.training.batch_size, num_workers=0
    )

    module = FettTranslationModule(
        base_model_checkpoint=cfg.model.base_model_checkpoint,
        head_cfg=dict(cfg.model.head),
        training_cfg=cfg.training,
    )

    callbacks = [
        EarlyStopping(
            monitor=cfg.training.callbacks.early_stopping.monitor,
            patience=cfg.training.callbacks.early_stopping.patience,
            mode=cfg.training.callbacks.early_stopping.mode,
        ),
        ModelCheckpoint(
            monitor=cfg.training.callbacks.model_checkpoint.monitor,
            mode=cfg.training.callbacks.model_checkpoint.mode,
            save_top_k=cfg.training.callbacks.model_checkpoint.save_top_k,
            filename="best_translation",
            dirpath=cfg.training.callbacks.model_checkpoint.dirpath,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        accelerator=cfg.training.trainer.accelerator,
        gradient_clip_val=cfg.training.trainer.gradient_clip_val,
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        callbacks=callbacks,
    )

    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(module, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
