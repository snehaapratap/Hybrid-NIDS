# models/gan.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int, n_feats: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_feats),
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    """
    Semi‑supervised: two heads
      • adv – real / fake (sigmoid/BCE)
      • cls – normal / unknown   (Softmax over 1 real class)
    """
    def __init__(self, n_feats: int, n_classes: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.adv = nn.Linear(64, 1)
        self.cls = nn.Linear(64, n_classes)

    def forward(self, x):
        h = self.body(x)
        return self.adv(h), self.cls(h)
