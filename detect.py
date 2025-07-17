# detect.py
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from models.gan import Discriminator
from models.vae import VAE_PT

# --- Config --------------------------------------------------
CSV_PATH = "data/IoT_Intrusion.csv"
GAN_DISC_PATH = "models/gan_discriminator.pt"
GAN_SCALER_PATH = "models/gan_scaler.pth"
VAE_PATH = "models/vae_final.pt"
THR_VAE = 0.015
THR_GAN = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Data -----------------------------------------------
df = pd.read_csv(CSV_PATH)

# Load GAN scaler info
gan_scaler = torch.load(GAN_SCALER_PATH, weights_only=False)
numeric_cols = gan_scaler["numeric_cols"]
scaler = StandardScaler()
scaler.mean_ = gan_scaler["scaler_mean"]
scaler.scale_ = gan_scaler["scaler_scale"]

X_all = scaler.transform(df[numeric_cols])

# Load the feature list used for VAE
with open("models/vae_features.txt") as f:
    vae_features = [line.strip() for line in f if line.strip()]

X_vae = df[vae_features].values
input_dim = len(vae_features)
latent_dim = 8  # Must match training

# --- Load VAE (PyTorch) --------------------------------------
from collections import OrderedDict
vae = VAE_PT(input_dim, latent_dim).to(DEVICE)
vae_state = torch.load(VAE_PATH, map_location=DEVICE)
if isinstance(vae_state, OrderedDict):
    vae.load_state_dict(vae_state, strict=False)
else:
    vae.load_state_dict(vae_state, strict=False)
vae.eval()

# --- Load Discriminator --------------------------------------
disc = Discriminator(len(numeric_cols)).to(DEVICE)
disc.load_state_dict(torch.load(GAN_DISC_PATH, map_location=DEVICE))
disc.eval()

# --- Detect Anomalies ----------------------------------------
anomalies = []
original_rows = []
attack_types = []

print("🔍 Running VAE + GAN-based detection...")

# 1. Batch VAE predictions (PyTorch)
with torch.no_grad():
    X_vae_tensor = torch.tensor(X_vae, dtype=torch.float32, device=DEVICE)
    vae_recon = vae.reconstruct(X_vae_tensor).cpu().numpy()
recon_errors = np.mean((vae_recon - X_vae) ** 2, axis=1)

# 2. Only check samples with high VAE error
for idx in range(len(X_vae)):
    if recon_errors[idx] < THR_VAE:
        continue  # not anomalous

    x_gan_np = X_all[idx]
    x_tensor = torch.tensor(x_gan_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # GAN Discriminator
    with torch.no_grad():
        adv_score, cls_prob = disc(x_tensor)
        adv_prob = torch.sigmoid(adv_score).item()

    if adv_prob < THR_GAN:
        anomalies.append(X_vae[idx])
        original_rows.append(df.iloc[idx])
        attack_types.append(df.iloc[idx].get('label', 'unknown'))

    if idx % 1000 == 0 and idx > 0:
        print(f"Processed {idx} samples...")

# --- Clustering ----------------------------------------------
def run_clustering(anomaly_data, method="DBSCAN", **kwargs):
    if method == "KMeans":
        n_clusters = kwargs.get("n_clusters", 3)
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    else:  # Default: DBSCAN
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(anomaly_data)
    return labels

if not anomalies:
    print("✅ No confirmed anomalies found.")
    exit()

print(f"🚨 {len(anomalies)} anomalies confirmed. Clustering...")

# Choose clustering method and parameters (can be exposed in UI)
clustering_method = "DBSCAN"  # or "KMeans"
clustering_params = {"eps": 0.5, "min_samples": 5} if clustering_method == "DBSCAN" else {"n_clusters": 3}

X_anom = StandardScaler().fit_transform(anomalies)
labels = run_clustering(X_anom, method=clustering_method, **clustering_params)

# --- Save Anomalies with Cluster Labels and Attack Types ------
clustered_df = pd.DataFrame(original_rows)
clustered_df["cluster"] = labels
if attack_types:
    clustered_df["attack_type"] = attack_types
clustered_df.to_csv("data/detected_anomalies_clustered.csv", index=False)

print(f"✅ Done. Saved to data/detected_anomalies_clustered.csv")
print("Detected Clusters:", set(labels))

# Print cluster summaries
print("\nCluster Summary:")
for cluster_id in sorted(set(labels)):
    count = (labels == cluster_id).sum()
    print(f"  Cluster {cluster_id}: {count} samples")
    # Optionally, print basic stats for each cluster
    if count > 0:
        cluster_samples = clustered_df[clustered_df["cluster"] == cluster_id]
        print(cluster_samples.describe(include='all').T)
        print()

# Print attack type summary
if attack_types:
    from collections import Counter
    print("Attack type counts among detected anomalies:")
    for attack, count in Counter(attack_types).items():
        print(f"  {attack}: {count}")
