# detect.py
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from models.gan import Discriminator
from tensorflow.keras.models import load_model
from models.vae import VAE  # must be defined before loading


# --- Config --------------------------------------------------
CSV_PATH = "data/combined_dataset.csv"
GAN_MODEL_PATH = "models/ss_gan_dns.pth"
VAE_MODEL_PATH = "models/vae_final.keras"
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

# Load the feature list used for VAE
with open("models/vae_features.txt") as f:
    vae_features = [line.strip() for line in f if line.strip()]

X_vae = df[vae_features].values
input_dim = len(vae_features)
latent_dim = 8  # Must match training

vae = VAE(input_dim, latent_dim)
vae(np.zeros((1, input_dim), dtype=np.float32))  # Build the model
vae.load_weights("models/vae_final.weights.h5")

# --- Load Discriminator --------------------------------------
disc = Discriminator(len(numeric_cols)).to(DEVICE)
disc.load_state_dict(gan_ckpt["disc"])
disc.eval()

# --- Detect Anomalies ----------------------------------------
anomalies = []
original_rows = []
attack_types = []

print("🔍 Running VAE + GAN-based detection...")
for idx in range(len(X_vae)):
    x_vae_np = X_vae[idx]  # 12 features for VAE
    x_gan_np = X_all[idx]  # 30 features for GAN
    x_tensor = torch.tensor(x_gan_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 1. VAE reconstruction
    x_recon = vae.predict(np.expand_dims(x_vae_np, 0), verbose=0)
    rec_err = np.mean((x_recon - x_vae_np) ** 2)

    if rec_err < THR_VAE:
        continue  # not anomalous

    # 2. GAN Discriminator
    with torch.no_grad():
        adv_score, cls_prob = disc(x_tensor)
        adv_prob = torch.sigmoid(adv_score).item()

    if adv_prob < THR_GAN:
        anomalies.append(x_vae_np)
        original_rows.append(df.iloc[idx])
        # Get attack type from 'label' column if present
        attack_types.append(df.iloc[idx].get('label', 'unknown'))

# --- Clustering ----------------------------------------------
if not anomalies:
    print("✅ No confirmed anomalies found.")
    exit()

print(f"🚨 {len(anomalies)} anomalies confirmed. Clustering...")

X_anom = StandardScaler().fit_transform(anomalies)
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_anom)

# --- Save Anomalies with Cluster Labels and Attack Types ------
clustered_df = pd.DataFrame(original_rows)
clustered_df["cluster"] = labels
if attack_types:
    clustered_df["attack_type"] = attack_types
clustered_df.to_csv("data/detected_anomalies_clustered.csv", index=False)

print(f"✅ Done. Saved to data/detected_anomalies_clustered.csv")
print("Detected Clusters:", set(labels))

# Print attack type summary
if attack_types:
    from collections import Counter
    print("Attack type counts among detected anomalies:")
    for attack, count in Counter(attack_types).items():
        print(f"  {attack}: {count}")
