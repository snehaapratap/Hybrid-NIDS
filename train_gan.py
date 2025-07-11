# train_gan.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import FlowDS
from models.gan import Gen, Disc, LATENT
from sklearn.preprocessing import StandardScaler
import pandas as pd

CSV = "data/stateless_features-light_image.pcap.csv"
BATCH = 256
EPOCHS = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- data prep -----------------------------------------------------------
df = pd.read_csv(CSV)
scaler = StandardScaler().fit(df.drop(columns=['label']))
train_ds = FlowDS(CSV, scaler, train=True)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

G, D = Gen().to(device), Disc().to(device)
opt_G = optim.Adam(G.parameters(), 2e-4)
opt_D = optim.Adam(D.parameters(), 2e-4)
bce = nn.BCEWithLogitsLoss()
ce  = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for x, t in train_loader:
        x, t = x.to(device), t.to(device)

        # ------------------ Train Discriminator ------------------
        z = torch.randn(len(x), LATENT, device=device)
        fake = G(z)
        D_real_adv, D_real_cls = D(x)
        D_fake_adv, _ = D(fake.detach())

        # labels
        y_real = torch.ones_like(D_real_adv)
        y_fake = torch.zeros_like(D_fake_adv)

        loss_adv = bce(D_real_adv, y_real) + bce(D_fake_adv, y_fake)
        # semi‑sup classification loss (only where t != -1)
        mask = (t != -1)
        loss_cls = ce(D_real_cls[mask], t[mask]) if mask.any() else 0.0
        loss_D = loss_adv + loss_cls

        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # ------------------ Train Generator ----------------------
        D_fake_adv, _ = D(fake)
        loss_G = bce(D_fake_adv, y_real)     # want fake → real
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | D: {loss_D.item():.3f} | G: {loss_G.item():.3f}")

# save
torch.save(G.state_dict(), "models/gen.pth")
torch.save(D.state_dict(), "models/disc.pth")
