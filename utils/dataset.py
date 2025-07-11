# utils/dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset

class FlowDS(Dataset):
    def __init__(self, csv_path, scaler=None, train=True):
        df = pd.read_csv(csv_path)
        self.y = df['label'].values          # string labels
        self.X = df.drop(columns=['label']).values.astype('float32')
        if scaler is not None:
            self.X = scaler.transform(self.X)
        self.train = train

        # semi‑sup: keep normal flows as class 0, others as ‑1 (unlabeled)
        self.targets = (self.y == 'normal').astype(int)
        if not train:
            self.targets[:] = -1             # unlabeled during GAN train

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        t = torch.tensor(self.targets[idx])
        return x, t
