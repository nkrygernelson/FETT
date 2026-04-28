from fett.data.data import FettDataset
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    train_set = FettDataset(cfg.data.processed_data_dir, split="train", target_col=cfg.data.target_col)
    val_set = FettDataset(cfg.data.processed_data_dir, split="val", target_col=cfg.data.target_col)
    test_set = FettDataset(cfg.data.processed_data_dir, split="test", target_col=cfg.data.target_col)

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.training.batch_size, num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.training.batch_size, num_workers=0
    )

    # Extract model architecture params as a plain dict so they can be embedded in checkpoints.
    # This allows the checkpoint to be self-contained — the model can be reconstructed even if
    # the config files change later.
    model_cfg = OmegaConf.to_container(cfg.model.model, resolve=True)
    model_cfg.pop("_target_", None)

    # Instantiate model, passing training config and model_cfg separately
    model = instantiate(cfg.model, training_cfg=cfg.training, model_cfg=model_cfg)

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
            filename=cfg.training.callbacks.model_checkpoint.filename,
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

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
