"""
Train the FidelityTranslationHead on top of a frozen pre-trained base model.

The base checkpoint must have been saved by FettLightningModule (with embedded model_cfg).

Usage:
    # Build the matched-pairs dataset first, e.g.
    uv run invoke make-data-pbesol-to-all

    # Single train/val/test split:
    uv run invoke train-translation --base-ckpt models/best.ckpt

    # 5-fold cross-validation (paper Table 4):
    uv run src/fett/train_translation.py model.base_model_checkpoint=models/best.ckpt \
        data=homemade_pbesol_to_all training.cv_folds=5
"""
import json
import logging
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader

from fett.callbacks import EpochPrint
from fett.data.data import FettTranslationDataset
from fett.model.lightning_module import FettTranslationModule
from fett.model.model_io import load_model_from_checkpoint


def _build_logger(cfg: DictConfig, run_suffix: str = ""):
    wb = cfg.training.get("logger", {}).get("wandb", {})
    if not wb or not wb.get("enabled", False):
        return True
    from lightning.pytorch.loggers import WandbLogger
    name = wb.get("name")
    if name and run_suffix:
        name = f"{name}-{run_suffix}"
    return WandbLogger(
        project=wb.get("project", "fett"),
        name=name,
        tags=list(wb.get("tags", [])) or None,
    )

log = logging.getLogger(__name__)


def _make_loaders(processed_dir: str, batch_size: int):
    train_set = FettTranslationDataset(processed_dir, split="train")
    val_set = FettTranslationDataset(processed_dir, split="val")
    test_set = FettTranslationDataset(processed_dir, split="test")
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_set, batch_size=batch_size, num_workers=0),
        DataLoader(test_set, batch_size=batch_size, num_workers=0),
        train_set,
        val_set,
        test_set,
    )


def _build_module(cfg: DictConfig, base_ckpt: str) -> FettTranslationModule:
    base_module = load_model_from_checkpoint(base_ckpt)
    module = FettTranslationModule(
        base_model_cfg=base_module.model_cfg,
        head_cfg=dict(cfg.model.head),
        training_cfg=cfg.training,
    )
    module.load_base_weights(base_ckpt)
    return module


def _build_trainer(cfg: DictConfig, ckpt_filename: str, run_suffix: str = "") -> L.Trainer:
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
            filename=ckpt_filename,
            dirpath=cfg.training.callbacks.model_checkpoint.dirpath,
        ),
    ]
    return L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        accelerator=cfg.training.trainer.accelerator,
        gradient_clip_val=cfg.training.trainer.gradient_clip_val,
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        enable_progress_bar=cfg.training.trainer.get("enable_progress_bar", False),
        callbacks=callbacks,
        logger=_build_logger(cfg, run_suffix),
    )


def _evaluate_module(module: FettTranslationModule, dataset: FettTranslationDataset, batch_size: int):
    """Run inference on a dataset and return denormalized metrics."""
    device = next(module.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size)
    preds, targets = [], []
    module.eval()
    with torch.no_grad():
        for element_ids, element_weights, src_fid, tgt_fid, src_bg, tgt_bg in loader:
            p = module(
                element_ids.to(device), element_weights.to(device),
                src_fid.to(device), tgt_fid.to(device), src_bg.to(device),
            )
            preds.append(p.cpu().numpy())
            targets.append(tgt_bg.numpy())
    preds = np.concatenate(preds) * dataset.tgt_std + dataset.tgt_mean
    targets = np.concatenate(targets) * dataset.tgt_std + dataset.tgt_mean
    return {
        "mae": float(mean_absolute_error(targets, preds)),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "r2": float(r2_score(targets, preds)),
    }


def _run_single(cfg: DictConfig) -> None:
    train_dl, val_dl, test_dl, *_ = _make_loaders(cfg.data.processed_data_dir, cfg.training.batch_size)
    module = _build_module(cfg, cfg.model.base_model_checkpoint)
    trainer = _build_trainer(cfg, "best_translation")
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.test(module, dataloaders=test_dl)


def _run_cv(cfg: DictConfig, k: int) -> None:
    """K-fold CV by formula. Each fold's train+val rows have no formula overlap with the
    fold's test rows, and BG normalization stats are recomputed per-fold from fold-train only."""
    full = pd.concat([
        pd.read_csv(Path(cfg.data.processed_data_dir) / f"{split}.csv")
        for split in ("train", "val", "test")
    ], ignore_index=True).reset_index(drop=True)

    cv_dir = Path(cfg.data.processed_data_dir) / "_cv"
    cv_dir.mkdir(exist_ok=True)

    gkf = GroupKFold(n_splits=k)
    folds = list(gkf.split(full, groups=full["formula"]))

    metrics_all = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(folds):
        log.info(f"=== Fold {fold_idx + 1}/{k} ===")

        train_val_df = full.iloc[train_val_idx]
        test_df = full.iloc[test_idx]

        # Group-aware val carve-out from train_val so val formulas don't bleed into train.
        gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=42 + fold_idx)
        tr_idx, va_idx = next(gss.split(train_val_df, groups=train_val_df["formula"]))
        train_df = train_val_df.iloc[tr_idx]
        val_df = train_val_df.iloc[va_idx]

        # Sanity: no formula leaks across train / val / test for this fold.
        tr_f, va_f, te_f = set(train_df["formula"]), set(val_df["formula"]), set(test_df["formula"])
        assert not (tr_f & va_f) and not (tr_f & te_f) and not (va_f & te_f), \
            "CV fold leaked formulas across splits"

        # Per-fold normalization stats — computed on TRAIN ONLY.
        stats = {
            "mean": float(train_df["target_bg"].mean()),
            "std": float(train_df["target_bg"].std()),
            "source_mean": float(train_df["source_bg"].mean()),
            "source_std": float(train_df["source_bg"].std()),
        }
        (cv_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
            df.to_csv(cv_dir / f"{name}.csv", index=False)

        train_set = FettTranslationDataset(str(cv_dir), split="train")
        val_set = FettTranslationDataset(str(cv_dir), split="val")
        test_set = FettTranslationDataset(str(cv_dir), split="test")

        train_dl = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True)
        val_dl = DataLoader(val_set, batch_size=cfg.training.batch_size)

        module = _build_module(cfg, cfg.model.base_model_checkpoint)
        trainer = _build_trainer(cfg, f"best_translation_fold{fold_idx + 1}", run_suffix=f"fold{fold_idx + 1}")
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
        fold_metrics = _evaluate_module(module, test_set, cfg.training.batch_size)
        log.info(f"Fold {fold_idx + 1}: {fold_metrics}")
        metrics_all.append(fold_metrics)

    summary = {
        m: {
            "mean": float(np.mean([x[m] for x in metrics_all])),
            "std": float(np.std([x[m] for x in metrics_all])),
        }
        for m in ("mae", "rmse", "r2")
    }
    out_path = cv_dir / "cv_summary.json"
    with open(out_path, "w") as f:
        json.dump({"folds": metrics_all, "summary": summary}, f, indent=2)
    log.info(f"\n5-fold CV summary written to {out_path}\n{json.dumps(summary, indent=2)}")


@hydra.main(config_path="../../configs", config_name="config_translation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cv_folds = int(cfg.training.get("cv_folds", 1) or 1)
    if cv_folds > 1:
        _run_cv(cfg, cv_folds)
    else:
        _run_single(cfg)


if __name__ == "__main__":
    main()
