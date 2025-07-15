# train_gan.py
"""
Train a Semi‑Supervised GAN on combined_dataset.csv
--------------------------------------------------
* normal_labels  – strings in df['label'] considered BENIGN
* everything else is left 'unlabeled' for the cls‑head
--------------------------------------------------
Run:
    python train_gan.py
"""

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.dataset import FlowDataset
from models.gan import Generator, Discriminator

CSV_PATH = "data/combined_dataset.csv"
NORMAL_LABELS = [
    "audio", "image", "text", "video", "compressed", "exe",
    # add / remove as you decide what is benign
]
LATENT_DIM = 100
BATCH = 256
EPOCHS = 40
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------
# 1)  Detect pure‑numeric columns & fit scaler
# --------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# discard obvious non‑numeric cols
DROP = {"label", "labels", "timestamp", "rr_type", "subdomain",
        "longest_word", "sld"}
numeric_cols = [
    c for c in df.columns
    if c not in DROP and pd.api.types.is_numeric_dtype(df[c])
]

assert numeric_cols, "❌ No numeric columns found – check the CSV."
print(f"✅ Numeric feature count: {len(numeric_cols)}")

scaler = StandardScaler().fit(df[numeric_cols])

# --------------------------------------------------------------------
# 2)  Data loader
# --------------------------------------------------------------------
ds = FlowDataset(
    CSV_PATH,
    numeric_cols,
    NORMAL_LABELS,
    scaler,
    device=DEVICE,
)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)

# --------------------------------------------------------------------
# 3)  Build models
# --------------------------------------------------------------------
G = Generator(LATENT_DIM, len(numeric_cols)).to(DEVICE)
D = Discriminator(len(numeric_cols)).to(DEVICE)
opt_G = optim.Adam(G.parameters(), lr=LR)
opt_D = optim.Adam(D.parameters(), lr=LR)
bce = nn.BCEWithLogitsLoss()
ce  = nn.CrossEntropyLoss()

# --------------------------------------------------------------------
# 4)  Training loop
# --------------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    for x, t in loader:
        # ------------------------------------------------- D step
        z = torch.randn(x.size(0), LATENT_DIM, device=DEVICE)
        fake = G(z).detach()          # stop grad for D step

        adv_real, cls_real = D(x)
        adv_fake, _        = D(fake)

        # adversarial loss
        y_real = torch.ones_like(adv_real)
        y_fake = torch.zeros_like(adv_fake)
        loss_adv = bce(adv_real, y_real) + bce(adv_fake, y_fake)

        # classification loss (mask unlabeled)
        mask = t != -1
        if mask.any():
            loss_cls = ce(cls_real[mask], t[mask])
        else:
            loss_cls = torch.tensor(0.0, device=DEVICE)

        loss_D = loss_adv + loss_cls
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # ------------------------------------------------- G step
        z = torch.randn(x.size(0), LATENT_DIM, device=DEVICE)
        fake = G(z)
        adv_fake, _ = D(fake)
        loss_G = bce(adv_fake, y_real)   # gen wants adv_fake → real
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    print(f"Epoch {epoch:02d}/{EPOCHS} "
          f"| D: {loss_D.item():.3f}  G: {loss_G.item():.3f}")

# --------------------------------------------------------------------
# 5)  Save weights
# --------------------------------------------------------------------
torch.save({
    "gen":  G.state_dict(),
    "disc": D.state_dict(),
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "numeric_cols": numeric_cols,
}, "models/ss_gan_dns.pth")

print("✅  Semi‑Supervised GAN trained and saved to models/ss_gan_dns.pth")
