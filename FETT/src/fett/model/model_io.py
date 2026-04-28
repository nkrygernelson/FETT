"""
Utilities for saving and loading FETT models with their architecture config embedded.

The model architecture config is stored inside the Lightning checkpoint under the
"model_cfg" key, so models can always be reconstructed even if the current Hydra
config files have changed.
"""
import torch
from pathlib import Path


def load_model_from_checkpoint(ckpt_path: str):
    """
    Load a FettLightningModule from a checkpoint that contains an embedded model_cfg.

    The checkpoint must have been saved by FettLightningModule (which embeds model_cfg
    in on_save_checkpoint). This lets you reconstruct the exact architecture that was
    used during training, independent of the current config files.

    Args:
        ckpt_path: Path to the .ckpt file.

    Returns:
        FettLightningModule with weights loaded, in eval mode.

    Raises:
        KeyError: If the checkpoint does not contain a 'model_cfg' key.
        FileNotFoundError: If ckpt_path does not exist.
    """
    from fett.model.lightning_module import FettLightningModule
    from fett.model.set_based_model import SetBasedBandgapModel
    from omegaconf import OmegaConf

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_cfg" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {ckpt_path} does not contain 'model_cfg'. "
            "It may have been saved before model config embedding was added. "
            "Re-train to generate a compatible checkpoint."
        )

    model_cfg: dict = checkpoint["model_cfg"]

    # Reconstruct the raw PyTorch model from the embedded config
    pytorch_model = SetBasedBandgapModel(
        num_elements=model_cfg.get("num_elements", 118),
        num_fidelities=model_cfg.get("num_fidelities", 5),
        embedding_dim=model_cfg.get("embedding_dim", 128),
        fidelity_dim=model_cfg.get("fidelity_dim", 16),
        num_blocks=model_cfg.get("num_blocks", 3),
        num_heads=model_cfg.get("num_heads", 4),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        dropout=model_cfg.get("dropout", 0.1),
        pooling_type=model_cfg.get("pooling_type", "weighted"),
        pooling_params=model_cfg.get("pooling_params", None),
    )

    # Reconstruct training_cfg stub (only needed for configure_optimizers, not for inference)
    training_cfg = OmegaConf.create(checkpoint.get("hyper_parameters", {}).get("training_cfg", {}))

    module = FettLightningModule(
        model=pytorch_model,
        training_cfg=training_cfg,
        model_cfg=model_cfg,
    )
    module.load_state_dict(checkpoint["state_dict"])
    module.eval()
    return module
