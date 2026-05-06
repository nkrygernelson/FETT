import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import hydra

from fett.model.set_based_model import SetBasedBandgapModel
from fett.model.translation_layer import FidelityTranslationHead


class FettLightningModule(L.LightningModule):
    """Lightning wrapper around the SetBasedBandgapModel for the base task."""

    def __init__(self, model: torch.nn.Module, training_cfg: DictConfig, model_cfg: dict | None = None):
        super().__init__()
        self.training_cfg = training_cfg
        self.model_cfg = dict(model_cfg) if model_cfg else {}
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["model_cfg"] = self.model_cfg

    def forward(self, element_ids, fidelities, element_weights=None, return_embedding: bool = False):
        return self.model(element_ids, fidelities, element_weights, return_embedding=return_embedding)

    def _step(self, batch, stage: str):
        element_ids, element_weights, fidelity, target = batch
        preds = self.model(element_ids, fidelity, element_weights)
        loss = F.mse_loss(preds, target)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=(stage != "test"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.training_cfg.optimizer, params=self.parameters())


class FettTranslationModule(L.LightningModule):
    """
    Translation pipeline. The frozen base model produces pooled embeddings at the
    source and target fidelity for the same compound; these are concatenated with the
    normalized source bandgap and fed through a small MLP to predict the target bandgap.

    Reload-friendly: the base-model architecture config is stored in ``hparams`` so the
    translation checkpoint can be reloaded without the original base checkpoint on disk
    (the base weights are restored via the saved ``state_dict``).
    """

    def __init__(
        self,
        base_model_cfg: dict,
        head_cfg: dict,
        training_cfg,
    ):
        super().__init__()
        if isinstance(training_cfg, DictConfig):
            training_cfg_dict = OmegaConf.to_container(training_cfg, resolve=True)
        else:
            training_cfg_dict = dict(training_cfg) if training_cfg else {}
        self.save_hyperparameters({
            "base_model_cfg": dict(base_model_cfg),
            "head_cfg": dict(head_cfg),
            "training_cfg": training_cfg_dict,
        })
        self.training_cfg = OmegaConf.create(training_cfg_dict)

        self.base_model = SetBasedBandgapModel(
            num_elements=base_model_cfg.get("num_elements", 118),
            num_fidelities=base_model_cfg.get("num_fidelities", 6),
            embedding_dim=base_model_cfg.get("embedding_dim", 128),
            fidelity_dim=base_model_cfg.get("fidelity_dim", 16),
            num_blocks=base_model_cfg.get("num_blocks", 3),
            num_heads=base_model_cfg.get("num_heads", 4),
            hidden_dim=base_model_cfg.get("hidden_dim", 128),
            dropout=base_model_cfg.get("dropout", 0.1),
            pooling_type=base_model_cfg.get("pooling_type", "weighted"),
            pooling_params=base_model_cfg.get("pooling_params", None),
        )
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

        embed_dim = (
            self.base_model.element_embedding.embedding.embedding_dim
            + self.base_model.fidelity_embedding.embedding.embedding_dim
        )
        self.head = FidelityTranslationHead(
            embed_dim=embed_dim,
            hidden_dim=head_cfg.get("hidden_dim", 128),
            dropout=head_cfg.get("dropout", 0.1),
        )

    def load_base_weights(self, base_ckpt_path: str) -> None:
        """Bootstrap the frozen base model from a FettLightningModule checkpoint."""
        ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
        prefix = "model."
        base_sd = {
            k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)
        }
        self.base_model.load_state_dict(base_sd)
        self.base_model.eval()

    def forward(self, element_ids, element_weights, source_fidelity, target_fidelity, source_bg):
        with torch.no_grad():
            src_emb = self.base_model(element_ids, source_fidelity, element_weights, return_embedding=True)
            tgt_emb = self.base_model(element_ids, target_fidelity, element_weights, return_embedding=True)
        return self.head(src_emb, tgt_emb, source_bg)

    def _step(self, batch, stage: str):
        element_ids, element_weights, src_fid, tgt_fid, src_bg, tgt_bg = batch
        preds = self(element_ids, element_weights, src_fid, tgt_fid, src_bg)
        loss = F.mse_loss(preds, tgt_bg)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=(stage != "test"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.training_cfg.optimizer, params=self.head.parameters())
