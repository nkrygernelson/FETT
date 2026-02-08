import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from fett.utils import formula_to_set_representation

log = logging.getLogger(__name__)

def prop_binning(df: pd.DataFrame, num_bins: int = 4, prop_name: str = "BG") -> pd.DataFrame:
    """
    Creates stratified bins for a column.
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

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def make_dataset(cfg: DictConfig) -> None:
    log.info(f"Processing data for: {cfg.data.name}")
    
    raw_dir = Path(cfg.data.raw_data_dir)
    processed_dir = Path(cfg.data.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    target_col = cfg.data.target_col
    split_ratios = cfg.data.split # [train, val, test]
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for fidelity_name, filename in cfg.data.files.items():
        fidelity_id = cfg.data.fidelity_map[fidelity_name]
        file_path = raw_dir / filename
        
        if not file_path.exists():
            log.warning(f"File not found: {file_path}. Skipping.")
            continue
            
        log.info(f"Loading {fidelity_name} from {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic Cleaning
        df = df.dropna(subset=["formula", target_col])
        df = df.drop_duplicates(subset=["formula"]) 
        
        # Validate Formulas
        # We check if formula_to_set_representation returns None
        valid_mask = df["formula"].apply(lambda x: formula_to_set_representation(x) is not None)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            log.warning(f"Dropping {invalid_count} invalid formulas from {fidelity_name}")
            df = df[valid_mask]
        
        # Add fidelity info
        df["fidelity_name"] = fidelity_name
        df["fidelity_id"] = fidelity_id
        
        # Stratified Split
        df = prop_binning(df, prop_name=target_col)
        
        train_size = split_ratios[0]
        test_val_size = 1.0 - train_size
        
        try:
            train_df, temp_df = train_test_split(
                df, train_size=train_size, stratify=df[f'{target_col}_category'], random_state=42
            )
            val_ratio = split_ratios[1] / test_val_size
            val_df, test_df = train_test_split(
                temp_df, train_size=val_ratio, stratify=temp_df[f'{target_col}_category'], random_state=42
            )
        except ValueError as e:
             log.warning(f"Splitting failed for {fidelity_name}: {e}. Using random split.")
             train_df, temp_df = train_test_split(df, train_size=train_size, random_state=42)
             val_ratio = split_ratios[1] / test_val_size
             val_df, test_df = train_test_split(temp_df, train_size=val_ratio, random_state=42)

        # Drop the category column before saving
        train_df = train_df.drop(columns=[f'{target_col}_category'])
        val_df = val_df.drop(columns=[f'{target_col}_category'])
        test_df = test_df.drop(columns=[f'{target_col}_category'])
        
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)
    
    # Combine
    full_train_df = pd.concat(train_dfs, ignore_index=True)
    full_val_df = pd.concat(val_dfs, ignore_index=True)
    full_test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Calculate Stats from TRAIN only
    mean = full_train_df[target_col].mean()
    std = full_train_df[target_col].std()
    
    stats = {"mean": float(mean), "std": float(std)}
    with open(processed_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    log.info(f"Stats (calculated from train set only) saved: {stats}")
    
    # Save as CSVs
    full_train_df.to_csv(processed_dir / "train.csv", index=False)
    log.info(f"Saved train.csv ({len(full_train_df)} samples)")
    
    full_val_df.to_csv(processed_dir / "val.csv", index=False)
    log.info(f"Saved val.csv ({len(full_val_df)} samples)")
    
    full_test_df.to_csv(processed_dir / "test.csv", index=False)
    log.info(f"Saved test.csv ({len(full_test_df)} samples)")

if __name__ == "__main__":
    make_dataset()