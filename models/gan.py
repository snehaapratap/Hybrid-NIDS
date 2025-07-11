# models/gan.py
import torch
import torch.nn as nn

LATENT = 100       # z‑dim
N_FEATS = 42       # ← set to feature count
N_CLASSES = 1      # only "normal" class + 1 fake label in semi‑sup

class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT, 128),
            nn.ReLU(),
            nn.Linear(128, N_FEATS),
        )
    def forward(self, z): return self.net(z)

class Disc(nn.Module):
    """
    Disc has two heads:
      • out_adv  – real/fake (W‑GAN style or BCE)
      • out_cls  – semi‑sup normal‑class probability
    """
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(N_FEATS, 128),
            nn.ReLU(),
        )
        self.out_adv = nn.Linear(128, 1)
        self.out_cls = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        h = self.body(x)
        return self.out_adv(h), self.out_cls(h)
