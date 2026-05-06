import torch
import json
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from fett.utils import formula_to_set_representation


class FettDataset(Dataset):
    """
    Dataset for the standard and only_new_on_expt modes.
    Each row: formula, fidelity_id, BG (normalized).
    Returns: (element_ids, element_weights, fidelity, target)
    """
    def __init__(self, processed_dir: str, split: str = "train", target_col: str = "BG", max_elements: int = 10):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.target_col = target_col
        self.max_elements = max_elements

        self.file_path = self.processed_dir / f"{split}.csv"
        self.stats_path = self.processed_dir / "stats.json"

        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.file_path}. Run 'uv run invoke make-data' first.")
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Stats file not found at {self.stats_path}. Run 'uv run invoke make-data' first.")

        self.df = pd.read_csv(self.file_path)

        with open(self.stats_path, "r") as f:
            self.stats = json.load(f)
        self.mean = self.stats["mean"]
        self.std = self.stats["std"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        """
        Returns:
            element_ids (Tensor): [max_elements]
            element_weights (Tensor): [max_elements]
            fidelity (Tensor): scalar long
            target (Tensor): scalar float (normalized bandgap)
        """
        row = self.df.iloc[index]
        formula = row["formula"]
        raw_target = row[self.target_col]
        fidelity = row["fidelity_id"]

        target = (raw_target - self.mean) / self.std

        res = formula_to_set_representation(formula, max_elements=self.max_elements)
        if res is None:
            element_ids = [1] + [0] * (self.max_elements - 1)
            element_weights = [1.0] + [0.0] * (self.max_elements - 1)
        else:
            element_ids, element_weights = res

        return (
            torch.tensor(element_ids, dtype=torch.long),
            torch.tensor(element_weights, dtype=torch.float32),
            torch.tensor(fidelity, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
        )


class FettTranslationDataset(Dataset):
    """
    Dataset for the matched-pairs translation mode.
    Each row: formula, source_fidelity_id, target_fidelity_id, source_bg, target_bg.
    Returns: (element_ids, element_weights, source_fidelity, target_fidelity, source_bg_norm, target_bg_norm)
    """
    def __init__(self, processed_dir: str, split: str = "train", max_elements: int = 10):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.max_elements = max_elements

        self.file_path = self.processed_dir / f"{split}.csv"
        self.stats_path = self.processed_dir / "stats.json"

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {self.file_path}. "
                "Run 'uv run invoke make-data-translation' first."
            )
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Stats file not found at {self.stats_path}.")

        self.df = pd.read_csv(self.file_path)

        with open(self.stats_path, "r") as f:
            stats = json.load(f)
        self.tgt_mean = stats["mean"]
        self.tgt_std = stats["std"]
        self.src_mean = stats.get("source_mean", stats["mean"])
        self.src_std = stats.get("source_std", stats["std"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        """
        Returns:
            element_ids (Tensor): [max_elements]
            element_weights (Tensor): [max_elements]
            source_fidelity (Tensor): scalar long
            target_fidelity (Tensor): scalar long
            source_bg (Tensor): scalar float (normalized)
            target_bg (Tensor): scalar float (normalized)
        """
        row = self.df.iloc[index]
        formula = row["formula"]
        src_fid = int(row["source_fidelity_id"])
        tgt_fid = int(row["target_fidelity_id"])
        src_bg = float(row["source_bg"])
        tgt_bg = float(row["target_bg"])

        src_bg_norm = (src_bg - self.src_mean) / self.src_std
        tgt_bg_norm = (tgt_bg - self.tgt_mean) / self.tgt_std

        res = formula_to_set_representation(formula, max_elements=self.max_elements)
        if res is None:
            element_ids = [1] + [0] * (self.max_elements - 1)
            element_weights = [1.0] + [0.0] * (self.max_elements - 1)
        else:
            element_ids, element_weights = res

        return (
            torch.tensor(element_ids, dtype=torch.long),
            torch.tensor(element_weights, dtype=torch.float32),
            torch.tensor(src_fid, dtype=torch.long),
            torch.tensor(tgt_fid, dtype=torch.long),
            torch.tensor(src_bg_norm, dtype=torch.float32),
            torch.tensor(tgt_bg_norm, dtype=torch.float32),
        )
