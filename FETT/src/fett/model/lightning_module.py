import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import hydra


class FettLightningModule(L.LightningModule):
    """
    Lightning wrapper for the SetBasedBandgapModel.
    Handles training, validation, optimization, and checkpoint config embedding.
    """
    def __init__(self, model: L.LightningModule, training_cfg: DictConfig, model_cfg: dict = None):
        super().__init__()
        self.training_cfg = training_cfg
        # model_cfg is stored so it can be embedded in checkpoints for future reconstruction
        self.model_cfg = model_cfg or {}
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Embed the model architecture config in every checkpoint."""
        checkpoint["model_cfg"] = self.model_cfg

    def forward(self, element_ids, fidelities, element_weights=None):
        """Delegates to the underlying PyTorch model."""
        return self.model(element_ids, fidelities, element_weights)

    def training_step(self, batch, batch_idx):
        element_ids, element_weights, fidelity, target = batch
        preds = self(element_ids, fidelity, element_weights)
        loss = F.mse_loss(preds, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        element_ids, element_weights, fidelity, target = batch
        preds = self(element_ids, fidelity, element_weights)
        loss = F.mse_loss(preds, target)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        element_ids, element_weights, fidelity, target = batch
        preds = self(element_ids, fidelity, element_weights)
        loss = F.mse_loss(preds, target)
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.training_cfg.optimizer, params=self.parameters())
        return optimizer


class FettTranslationModule(L.LightningModule):
    """
    Lightning module for the translation model.
    Loads a pre-trained frozen base model and trains a FidelityTranslationHead on top.
    Input batch: (element_ids, element_weights, source_fidelity, target_fidelity, source_bg, target_bg)
    """
    def __init__(self, base_model_checkpoint: str, head_cfg: dict, training_cfg):
        super().__init__()
        self.save_hyperparameters()
        # training_cfg may be a plain dict when restored from checkpoint hparams
        if isinstance(training_cfg, dict):
            training_cfg = OmegaConf.create(training_cfg)
        self.training_cfg = training_cfg

        # Load frozen base model
        from fett.model.model_io import load_model_from_checkpoint
        base_module = load_model_from_checkpoint(base_model_checkpoint)
        self.base_model = base_module.model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # Trainable translation head
        from fett.model.translation_layer import FidelityTranslationHead
        embed_dim = self.base_model.element_embedding.embedding.embedding_dim + \
                    self.base_model.fidelity_embedding.embedding.embedding_dim
        self.head = FidelityTranslationHead(
            embed_dim=embed_dim,
            fidelity_dim=self.base_model.fidelity_embedding.embedding.embedding_dim,
            num_fidelities=self.base_model.fidelity_embedding.embedding.num_embeddings,
            hidden_dim=head_cfg.get("hidden_dim", 128),
            dropout=head_cfg.get("dropout", 0.1),
        )

    def forward(self, element_ids, element_weights, source_fidelity, target_fidelity, source_bg):
        with torch.no_grad():
            embedding = self.base_model(
                element_ids, source_fidelity, element_weights, return_embedding=True
            )
        return self.head(embedding, target_fidelity, source_bg)

    def training_step(self, batch, batch_idx):
        element_ids, element_weights, src_fid, tgt_fid, src_bg, tgt_bg = batch
        preds = self(element_ids, element_weights, src_fid, tgt_fid, src_bg)
        loss = F.mse_loss(preds, tgt_bg)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        element_ids, element_weights, src_fid, tgt_fid, src_bg, tgt_bg = batch
        preds = self(element_ids, element_weights, src_fid, tgt_fid, src_bg)
        loss = F.mse_loss(preds, tgt_bg)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        element_ids, element_weights, src_fid, tgt_fid, src_bg, tgt_bg = batch
        preds = self(element_ids, element_weights, src_fid, tgt_fid, src_bg)
        loss = F.mse_loss(preds, tgt_bg)
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.training_cfg.optimizer, params=self.head.parameters()
        )
        return optimizer
