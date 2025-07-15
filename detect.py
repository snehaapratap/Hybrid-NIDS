# detect.py
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from models.gan import Discriminator
from tensorflow.keras.models import load_model
from models.vae import VAE  # must be defined before loading
from tensorflow.keras.models import load_model


# --- Config --------------------------------------------------
CSV_PATH = "data/combined_dataset.csv"
GAN_MODEL_PATH = "models/ss_gan_dns.pth"
VAE_MODEL_PATH = "models/vae_model.keras"
THR_VAE = 0.015
THR_GAN = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Data -----------------------------------------------
df = pd.read_csv(CSV_PATH)

# Load GAN model + scaler info
gan_ckpt = torch.load(GAN_MODEL_PATH, weights_only=False)
numeric_cols = gan_ckpt["numeric_cols"]
scaler = StandardScaler()
scaler.mean_ = gan_ckpt["scaler_mean"]
scaler.scale_ = gan_ckpt["scaler_scale"]

X_all = scaler.transform(df[numeric_cols])

# --- Load VAE -----------------------------------------------
vae = load_model(VAE_MODEL_PATH)

# --- Load Discriminator --------------------------------------
disc = Discriminator(len(numeric_cols)).to(DEVICE)
disc.load_state_dict(gan_ckpt["disc"])
disc.eval()

# --- Detect Anomalies ----------------------------------------
anomalies = []
original_rows = []

print("🔍 Running VAE + GAN-based detection...")
for idx in range(len(X_all)):
    x_np = X_all[idx]
    x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 1. VAE reconstruction
    x_recon = vae.predict(np.expand_dims(x_np, 0), verbose=0)
    rec_err = np.mean((x_recon - x_np) ** 2)

    if rec_err < THR_VAE:
        continue  # not anomalous

    # 2. GAN Discriminator
    with torch.no_grad():
        adv_score, cls_prob = disc(x_tensor)
        adv_prob = torch.sigmoid(adv_score).item()

    if adv_prob < THR_GAN:
        anomalies.append(x_np)
        original_rows.append(df.iloc[idx])

# --- Clustering ----------------------------------------------
if not anomalies:
    print("✅ No confirmed anomalies found.")
    exit()

print(f"🚨 {len(anomalies)} anomalies confirmed. Clustering...")

X_anom = StandardScaler().fit_transform(anomalies)
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_anom)

# --- Save Anomalies with Cluster Labels ----------------------
clustered_df = pd.DataFrame(original_rows)
clustered_df["cluster"] = labels
clustered_df.to_csv("data/detected_anomalies_clustered.csv", index=False)

print(f"✅ Done. Saved to data/detected_anomalies_clustered.csv")
print("Detected Clusters:", set(labels))
