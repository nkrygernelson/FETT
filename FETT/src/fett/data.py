import torch
import json
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from fett.utils import formula_to_set_representation

class FettDataset(Dataset):
    """
    Dataset class for loading CSV files and processing them on-the-fly.
    """
    def __init__(self, processed_dir: str, split: str = "train", target_col: str = "BG", max_elements: int = 10):
        """
        Args:
            processed_dir: Path to the directory containing {split}.csv files.
            split: One of 'train', 'val', 'test'.
            target_col: Name of the target column in the CSV.
            max_elements: Maximum number of elements for set representation.
        """
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
            
        # Load Data
        self.df = pd.read_csv(self.file_path)
        
        # Load Stats
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
            fidelity (Tensor): scalar
            target (Tensor): scalar (normalized bandgap)
        """
        row = self.df.iloc[index]
        formula = row["formula"]
        raw_target = row[self.target_col]
        fidelity = row["fidelity_id"] # Assuming this column exists from make_dataset
        
        # Normalize Target
        target = (raw_target - self.mean) / self.std
        
        # Process Formula
        # Note: In a production loop, caching this might be faster, but for now we follow the user's request.
        # Since we filtered bad formulas in make_dataset, this should not return None.
        # But for safety, we handle it or assume it's good.
        res = formula_to_set_representation(formula, max_elements=self.max_elements)
        if res is None:
             # Fallback (should have been filtered out)
             element_ids = [1] + [0] * (self.max_elements - 1)
             element_weights = [1.0] + [0.0] * (self.max_elements - 1)
        else:
            element_ids, element_weights = res
            
        return (
            torch.tensor(element_ids, dtype=torch.long),
            torch.tensor(element_weights, dtype=torch.float32),
            torch.tensor(fidelity, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32)
        )
