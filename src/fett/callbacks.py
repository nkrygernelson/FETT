"""Lightweight callbacks used by the training scripts."""
import lightning as L


class EpochPrint(L.Callback):
    """One-line per-epoch summary, replacement for the tqdm progress bar."""

    def on_train_epoch_end(self, trainer, _pl_module=None) -> None:  # noqa: ARG002
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        train = m.get("train/loss_epoch", m.get("train/loss"))
        val = m.get("val/loss")
        parts = [f"epoch {trainer.current_epoch + 1:>3d}/{trainer.max_epochs}"]
        if train is not None:
            parts.append(f"train_loss={float(train):.4f}")
        if val is not None:
            parts.append(f"val_loss={float(val):.4f}")
        print(" | ".join(parts), flush=True)
