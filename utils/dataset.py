# utils/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List

class FlowDataset(Dataset):
    """
    Returns (features_tensor, target)
      • features  – float32 vector
      • target    – 0 for normal, -1 for unlabeled
    """
    def __init__(
        self,
        csv_path: str,
        numeric_cols: List[str],
        normal_labels: List[str],
        scaler,
        device="cpu",
    ):
        df = pd.read_csv(csv_path)
        self.X = df[numeric_cols].astype("float32").values
        self.X = scaler.transform(self.X)
        raw_labels = df["label"].values.astype(str)
        self.targets = torch.tensor(
            [0 if l in normal_labels else -1 for l in raw_labels],
            dtype=torch.long,
        )
        self.device = device

    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        t = self.targets[idx]
        return x.to(self.device), t.to(self.device)
