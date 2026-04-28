"""
Evaluation scripts for the base and translation models.

Usage (base model):
    uv run invoke evaluate --ckpt models/best.ckpt
    # or directly:
    uv run src/fett/evaluate.py eval.checkpoint=models/best.ckpt

Usage (translation model):
    uv run invoke evaluate-translation --ckpt models/best_translation.ckpt
    # or directly:
    uv run src/fett/evaluate.py eval.checkpoint=models/best_translation.ckpt eval.mode=translation
"""
import json
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

log = logging.getLogger(__name__)


def _denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def _compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def _save_scatter(targets, predictions, metrics, label: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, predictions, alpha=0.4, s=10)
    lo = min(float(targets.min()), float(predictions.min()))
    hi = max(float(targets.max()), float(predictions.max()))
    ax.plot([lo, hi], [lo, hi], "r--", label="Ideal")
    ax.text(
        0.05, 0.95,
        f"MAE={metrics['mae']:.3f} eV\nRMSE={metrics['rmse']:.3f} eV\nR²={metrics['r2']:.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("Actual Bandgap (eV)")
    ax.set_ylabel("Predicted Bandgap (eV)")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    safe_label = label.replace("→", "_to_")
    fig.savefig(out_dir / f"{safe_label}_scatter.png", dpi=150)
    plt.close(fig)


def _save_summary(results: dict, out_dir: Path) -> None:
    labels = list(results.keys())
    maes = [results[f]["mae"] for f in labels]
    rmses = [results[f]["rmse"] for f in labels]
    r2s = [results[f]["r2"] for f in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w / 2, maes, w, label="MAE")
    ax1.bar(x + w / 2, rmses, w, label="RMSE")
    ax1.set_ylabel("Error (eV)")
    ax1.set_title("MAE and RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, r2s, color="green")
    ax2.set_ylabel("R²")
    ax2.set_title("R²")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylim(min(0, min(r2s) - 0.05), 1.05)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "performance_summary.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"label": labels, "mae": maes, "rmse": rmses, "r2": r2s}).to_csv(
        out_dir / "performance_summary.csv", index=False
    )
    log.info(f"Summary saved to {out_dir}")


def evaluate_base(cfg: DictConfig) -> None:
    """
    Evaluate the base FettLightningModule on the test split, broken down per fidelity.
    """
    from fett.model.model_io import load_model_from_checkpoint
    from fett.data.data import FettDataset

    ckpt_path = cfg.eval.checkpoint
    processed_dir = Path(cfg.data.processed_data_dir)
    out_dir = Path("reports") / "figures" / Path(ckpt_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading base model from {ckpt_path}")
    module = load_model_from_checkpoint(ckpt_path)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    module.to(device)
    module.eval()

    with open(processed_dir / "stats.json") as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    test_set = FettDataset(processed_dir, split="test", target_col=cfg.data.target_col)
    fidelity_map_inv = {v: k for k, v in cfg.data.fidelity_map.items()}
    results = {}

    for fid_id, fid_name in sorted(fidelity_map_inv.items()):
        mask = test_set.df["fidelity_id"] == fid_id
        fid_indices = [i for i, flag in enumerate(mask) if flag]
        if not fid_indices:
            log.warning(f"No test samples for fidelity {fid_name}. Skipping.")
            continue

        subset = tud.Subset(test_set, fid_indices)
        loader = tud.DataLoader(subset, batch_size=cfg.training.batch_size)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for element_ids, element_weights, fidelity, target in loader:
                element_ids = element_ids.to(device)
                element_weights = element_weights.to(device)
                fidelity = fidelity.to(device)
                preds = module(element_ids, fidelity, element_weights)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.numpy())

        preds_orig = _denormalize(np.array(all_preds), mean, std)
        tgts_orig = _denormalize(np.array(all_targets), mean, std)

        metrics = _compute_metrics(tgts_orig, preds_orig)
        results[fid_name] = metrics
        log.info(f"{fid_name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")

        pd.DataFrame({"actual": tgts_orig, "predicted": preds_orig}).to_csv(
            out_dir / f"{fid_name}_predictions.csv", index=False
        )
        _save_scatter(tgts_orig, preds_orig, metrics, fid_name, out_dir)

    if results:
        _save_summary(results, out_dir)

    print(f"\n{'Fidelity':<12} {'MAE (eV)':<12} {'RMSE (eV)':<12} {'R²':<8}")
    print("-" * 46)
    for fid, m in results.items():
        print(f"{fid:<12} {m['mae']:<12.4f} {m['rmse']:<12.4f} {m['r2']:<8.4f}")


def evaluate_translation(cfg: DictConfig) -> None:
    """
    Evaluate the FettTranslationModule on the translation test split, per fidelity pair.
    """
    from fett.model.lightning_module import FettTranslationModule
    from fett.data.data import FettTranslationDataset

    ckpt_path = cfg.eval.checkpoint
    processed_dir = Path(cfg.data.processed_data_dir)
    out_dir = Path("reports") / "figures" / Path(ckpt_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading translation model from {ckpt_path}")
    module = FettTranslationModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    module.to(device)
    module.eval()

    test_set = FettTranslationDataset(processed_dir, split="test")
    tgt_mean, tgt_std = test_set.tgt_mean, test_set.tgt_std

    fidelity_map_inv = {v: k for k, v in cfg.data.fidelity_map.items()}
    results = {}

    test_df = test_set.df.reset_index(drop=True)
    for (src_fid, tgt_fid), group in test_df.groupby(["source_fidelity_id", "target_fidelity_id"]):
        src_name = fidelity_map_inv.get(int(src_fid), str(src_fid))
        tgt_name = fidelity_map_inv.get(int(tgt_fid), str(tgt_fid))
        pair_name = f"{src_name}→{tgt_name}"

        indices = list(group.index)
        subset = tud.Subset(test_set, indices)
        loader = tud.DataLoader(subset, batch_size=cfg.training.batch_size)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for element_ids, element_weights, src_f, tgt_f, src_bg, tgt_bg in loader:
                element_ids = element_ids.to(device)
                element_weights = element_weights.to(device)
                src_f = src_f.to(device)
                tgt_f = tgt_f.to(device)
                src_bg = src_bg.to(device)
                preds = module(element_ids, element_weights, src_f, tgt_f, src_bg)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(tgt_bg.numpy())

        preds_orig = _denormalize(np.array(all_preds), tgt_mean, tgt_std)
        tgts_orig = _denormalize(np.array(all_targets), tgt_mean, tgt_std)

        metrics = _compute_metrics(tgts_orig, preds_orig)
        results[pair_name] = metrics
        log.info(f"{pair_name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")

        safe_name = pair_name.replace("→", "_to_")
        pd.DataFrame({"actual": tgts_orig, "predicted": preds_orig}).to_csv(
            out_dir / f"{safe_name}_predictions.csv", index=False
        )
        _save_scatter(tgts_orig, preds_orig, metrics, pair_name, out_dir)

    if results:
        _save_summary(results, out_dir)

    print(f"\n{'Pair':<22} {'MAE (eV)':<12} {'RMSE (eV)':<12} {'R²':<8}")
    print("-" * 56)
    for pair, m in results.items():
        print(f"{pair:<22} {m['mae']:<12.4f} {m['rmse']:<12.4f} {m['r2']:<8.4f}")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    mode = cfg.eval.mode
    if mode == "translation":
        evaluate_translation(cfg)
    else:
        evaluate_base(cfg)


if __name__ == "__main__":
    main()
