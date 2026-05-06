import logging
import json
from itertools import combinations
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from fett.utils import formula_to_set_representation

log = logging.getLogger(__name__)


def prop_binning(df: pd.DataFrame, num_bins: int = 4, prop_name: str = "BG") -> pd.DataFrame:
    """
    Creates stratified bins for a column, grouping outliers into the highest bin.
    """
    df = df.copy()
    if df[prop_name].isnull().any():
        log.warning(f"NaNs found in {prop_name}, dropping for binning calculation.")
        df = df.dropna(subset=[prop_name])

    Q1 = df[prop_name].quantile(0.25)
    Q3 = df[prop_name].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    is_outlier = df[prop_name] > outlier_threshold

    try:
        df[f'{prop_name}_category'] = pd.cut(
            df.loc[~is_outlier, prop_name],
            bins=num_bins,
            labels=False,
            duplicates='drop'
        )
    except ValueError as e:
        log.warning(f"Binning failed: {e}. Using simple quantile binning.")
        df[f'{prop_name}_category'] = pd.qcut(df[prop_name], q=num_bins, labels=False, duplicates='drop')

    highest_bin = df[f'{prop_name}_category'].max()
    if pd.isna(highest_bin):
        highest_bin = 0

    df[f'{prop_name}_category'] = df[f'{prop_name}_category'].fillna(highest_bin)
    return df


def _load_fidelity_dfs(cfg: DictConfig, target_col: str) -> dict:
    """Load, clean, and validate all fidelity CSVs. Returns dict keyed by fidelity_name."""
    raw_dir = Path(cfg.data.raw_data_dir)
    dfs = {}
    for fidelity_name, filename in cfg.data.files.items():
        fidelity_id = cfg.data.fidelity_map[fidelity_name]
        file_path = raw_dir / filename
        if not file_path.exists():
            log.warning(f"File not found: {file_path}. Skipping {fidelity_name}.")
            continue
        log.info(f"Loading {fidelity_name} from {file_path}")
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["formula", target_col])
        df = df.drop_duplicates(subset=["formula"])
        valid_mask = df["formula"].apply(lambda x: formula_to_set_representation(x) is not None)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            log.warning(f"Dropping {invalid_count} invalid formulas from {fidelity_name}")
            df = df[valid_mask]
        df["fidelity_name"] = fidelity_name
        df["fidelity_id"] = fidelity_id
        dfs[fidelity_name] = df
    return dfs


def _stratified_split(df: pd.DataFrame, split_ratios: list, target_col: str) -> tuple:
    """Split a DataFrame into train/val/test with stratification on target_col_category."""
    df = prop_binning(df, prop_name=target_col)
    train_size = split_ratios[0]
    test_val_size = 1.0 - train_size
    try:
        train_df, temp_df = train_test_split(
            df, train_size=train_size,
            stratify=df[f'{target_col}_category'], random_state=42
        )
        val_ratio = split_ratios[1] / test_val_size
        val_df, test_df = train_test_split(
            temp_df, train_size=val_ratio,
            stratify=temp_df[f'{target_col}_category'], random_state=42
        )
    except ValueError as e:
        log.warning(f"Stratified split failed: {e}. Using random split.")
        train_df, temp_df = train_test_split(df, train_size=train_size, random_state=42)
        val_ratio = split_ratios[1] / test_val_size
        val_df, test_df = train_test_split(temp_df, train_size=val_ratio, random_state=42)

    cat_col = f'{target_col}_category'
    for part in [train_df, val_df, test_df]:
        if cat_col in part.columns:
            part.drop(columns=[cat_col], inplace=True)
    return train_df, val_df, test_df


def _make_standard(cfg: DictConfig, target_col: str, processed_dir: Path) -> None:
    """Standard multi-fidelity dataset: each fidelity split independently, then combined."""
    dfs = _load_fidelity_dfs(cfg, target_col)
    split_ratios = cfg.data.split

    train_dfs, val_dfs, test_dfs = [], [], []
    for fidelity_name, df in dfs.items():
        tr, va, te = _stratified_split(df, split_ratios, target_col)
        train_dfs.append(tr)
        val_dfs.append(va)
        test_dfs.append(te)

    full_train = pd.concat(train_dfs, ignore_index=True)
    full_val = pd.concat(val_dfs, ignore_index=True)
    full_test = pd.concat(test_dfs, ignore_index=True)

    mean = full_train[target_col].mean()
    std = full_train[target_col].std()
    stats = {"mean": float(mean), "std": float(std)}
    with open(processed_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    log.info(f"Stats (train set only): {stats}")

    full_train.to_csv(processed_dir / "train.csv", index=False)
    full_val.to_csv(processed_dir / "val.csv", index=False)
    full_test.to_csv(processed_dir / "test.csv", index=False)
    log.info(f"Saved train={len(full_train)}, val={len(full_val)}, test={len(full_test)}")


def _make_only_new_on_expt(cfg: DictConfig, target_col: str, processed_dir: Path) -> None:
    """
    Dataset where the EXPT test set contains ONLY formulas not present in any lower-fidelity
    training data. Tests true generalization.

    Logic (from grave/dataset_creator.py:prepare_only_new_on_expt):
      1. Split EXPT first to get expt_test formulas.
      2. Remove those formulas from ALL lower-fidelity data.
      3. Split remaining lower-fidelity data normally.
      4. Combined train = lower-fidelity trains + expt_train.
    """
    dfs = _load_fidelity_dfs(cfg, target_col)
    split_ratios = cfg.data.split

    if "expt" not in dfs:
        raise ValueError("'expt' fidelity is required for only_new_on_expt mode.")

    # Step 1: Split EXPT first
    expt_train, expt_val, expt_test = _stratified_split(dfs["expt"], split_ratios, target_col)
    expt_test_formulas = set(expt_test["formula"])
    log.info(f"EXPT test set: {len(expt_test)} samples ({len(expt_test_formulas)} unique formulas)")

    train_dfs = [expt_train]
    val_dfs = [expt_val]
    test_dfs = [expt_test]

    # Step 2 & 3: Remove expt_test formulas from lower fidelities, then split
    for fidelity_name, df in dfs.items():
        if fidelity_name == "expt":
            continue
        before = len(df)
        df_filtered = df[~df["formula"].isin(expt_test_formulas)]
        removed = before - len(df_filtered)
        if removed > 0:
            log.info(f"Removed {removed} formula overlaps from {fidelity_name}")
        tr, va, te = _stratified_split(df_filtered, split_ratios, target_col)
        train_dfs.append(tr)
        val_dfs.append(va)
        test_dfs.append(te)

    full_train = pd.concat(train_dfs, ignore_index=True)
    full_val = pd.concat(val_dfs, ignore_index=True)
    full_test = pd.concat(test_dfs, ignore_index=True)

    mean = full_train[target_col].mean()
    std = full_train[target_col].std()
    stats = {"mean": float(mean), "std": float(std)}
    with open(processed_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    log.info(f"Stats (train set only): {stats}")

    full_train.to_csv(processed_dir / "train.csv", index=False)
    full_val.to_csv(processed_dir / "val.csv", index=False)
    full_test.to_csv(processed_dir / "test.csv", index=False)
    log.info(
        f"Saved train={len(full_train)}, val={len(full_val)}, test={len(full_test)}. "
        f"EXPT test has {len(expt_test)} samples with no formula overlap to lower-fidelity training."
    )


def _make_translation(
    cfg: DictConfig,
    target_col: str,
    processed_dir: Path,
    source_fidelity_ids: list | None = None,
    target_fidelity_ids: list | None = None,
) -> None:
    """
    Matched-pairs translation dataset.

    For each formula that appears at ≥2 fidelity levels, creates all ordered pairs
    (source_fidelity < target_fidelity). Row format:
        formula, source_fidelity_id, target_fidelity_id, source_bg, target_bg

    Args:
        source_fidelity_ids: If set, only keep pairs where source fidelity is in this list.
        target_fidelity_ids: If set, only keep pairs where target fidelity is in this list.
    """
    dfs = _load_fidelity_dfs(cfg, target_col)
    split_ratios = cfg.data.split

    # Build per-fidelity formula→BG lookup
    fidelity_lookup: dict[str, dict] = {}
    for fidelity_name, df in dfs.items():
        fidelity_lookup[fidelity_name] = {
            row["formula"]: (row[target_col], row["fidelity_id"])
            for _, row in df.iterrows()
        }

    all_formulas = set()
    for d in fidelity_lookup.values():
        all_formulas.update(d.keys())

    fidelity_names = list(dfs.keys())
    rows = []
    for formula in all_formulas:
        present = [
            (name, fidelity_lookup[name][formula])
            for name in fidelity_names
            if formula in fidelity_lookup[name]
        ]
        if len(present) < 2:
            continue
        # Sort by fidelity_id to create source < target ordered pairs
        present.sort(key=lambda x: x[1][1])
        for (src_name, (src_bg, src_fid)), (tgt_name, (tgt_bg, tgt_fid)) in combinations(present, 2):
            if source_fidelity_ids is not None and src_fid not in source_fidelity_ids:
                continue
            if target_fidelity_ids is not None and tgt_fid not in target_fidelity_ids:
                continue
            rows.append({
                "formula": formula,
                "source_fidelity_id": src_fid,
                "target_fidelity_id": tgt_fid,
                "source_bg": src_bg,
                "target_bg": tgt_bg,
            })

    if not rows:
        raise ValueError(
            f"No matched pairs found with source_fidelity_ids={source_fidelity_ids}, "
            f"target_fidelity_ids={target_fidelity_ids}. "
            "Check that multiple fidelities share formulas."
        )

    pairs_df = pd.DataFrame(rows).reset_index(drop=True)
    log.info(f"Created {len(pairs_df)} matched pairs from {len(all_formulas)} unique formulas")

    # Group-aware split keyed on formula: every pair sharing a formula must end up in the SAME
    # split. Otherwise the head sees (formula, source_bg, source_fid) at train time and is
    # asked to predict the same compound at a different target fidelity at test time, which
    # would leak the formula's identity through the frozen base-model embeddings.
    train_size = split_ratios[0]
    val_size = split_ratios[1]
    test_val_size = 1.0 - train_size

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    train_idx, temp_idx = next(gss1.split(pairs_df, groups=pairs_df["formula"]))
    train_df = pairs_df.iloc[train_idx].copy()
    temp_df = pairs_df.iloc[temp_idx].copy()

    val_ratio = val_size / test_val_size
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["formula"]))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Sanity check: no formula should appear in more than one split.
    train_f = set(train_df["formula"])
    val_f = set(val_df["formula"])
    test_f = set(test_df["formula"])
    overlap = (train_f & test_f) | (train_f & val_f) | (val_f & test_f)
    assert not overlap, f"Group split leaked {len(overlap)} formulas across splits"

    # Stats calculated from target_bg of training set only
    mean = train_df["target_bg"].mean()
    std = train_df["target_bg"].std()
    # Also store source_bg stats for normalization at load time
    src_mean = train_df["source_bg"].mean()
    src_std = train_df["source_bg"].std()
    stats = {
        "mean": float(mean),
        "std": float(std),
        "source_mean": float(src_mean),
        "source_std": float(src_std),
    }
    with open(processed_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    log.info(f"Translation stats (train only): {stats}")

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    log.info(f"Saved train={len(train_df)}, val={len(val_df)}, test={len(test_df)} pair rows")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def make_dataset(cfg: DictConfig) -> None:
    log.info(f"Processing data for: {cfg.data.name}")
    target_col = cfg.data.target_col
    processed_dir = Path(cfg.data.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_mode = cfg.data.get("dataset_mode", "standard")
    log.info(f"Dataset mode: {dataset_mode}")

    if dataset_mode == "standard":
        _make_standard(cfg, target_col, processed_dir)
    elif dataset_mode == "only_new_on_expt":
        _make_only_new_on_expt(cfg, target_col, processed_dir)
    elif dataset_mode == "translation":
        _make_translation(cfg, target_col, processed_dir)
    elif dataset_mode == "pbe_to_all":
        # PBE/GGA (fidelity 0) → any higher fidelity
        pbe_id = cfg.data.fidelity_map.get("pbe", 0)
        _make_translation(cfg, target_col, processed_dir, source_fidelity_ids=[pbe_id])
    elif dataset_mode == "pbesol_to_all":
        # PBEsol (fidelity 1) → any higher fidelity (the primary translation use-case)
        pbesol_id = cfg.data.fidelity_map.get("pbe-sol", 1)
        _make_translation(cfg, target_col, processed_dir, source_fidelity_ids=[pbesol_id])
    elif dataset_mode == "to_expt":
        # Any lower fidelity → EXPT (fidelity 5) only
        expt_id = cfg.data.fidelity_map.get("expt", 5)
        _make_translation(cfg, target_col, processed_dir, target_fidelity_ids=[expt_id])
    else:
        raise ValueError(
            f"Unknown dataset_mode '{dataset_mode}'. "
            "Choose from: standard, only_new_on_expt, translation, "
            "pbe_to_all, pbesol_to_all, to_expt"
        )


if __name__ == "__main__":
    make_dataset()
