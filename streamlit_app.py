import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from models.vae import VAE
from models.gan import Discriminator
import os

st.set_page_config(page_title="Hybrid NIDS", layout="wide")
st.title("💡 Hybrid NIDS: Autoencoder + GAN + Clustering")
st.markdown("""
**Modular, layered anomaly detection for network traffic.**
- Autoencoder: First-line anomaly detector
- GAN: Confirms true anomalies, filters false positives
- Clustering: Explains and groups threats for analysis
""")

# --- Sidebar: Data Selection ---
st.sidebar.header("Step 1: Data Selection")
data_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
default_path = "data/combined_dataset.csv"

if data_file is not None:
    df = pd.read_csv(data_file)
else:
    st.sidebar.info(f"Using default: {default_path}")
    df = pd.read_csv(default_path)

# --- Sidebar: Clustering Options ---
st.sidebar.header("Step 2: Clustering Options")
clustering_method = st.sidebar.selectbox("Clustering Algorithm", ["DBSCAN", "KMeans"])
if clustering_method == "DBSCAN":
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.5, 0.05)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5, 1)
    clustering_params = {"eps": eps, "min_samples": min_samples}
else:
    n_clusters = st.sidebar.slider("KMeans n_clusters", 2, 10, 3, 1)
    clustering_params = {"n_clusters": n_clusters}

# --- Sidebar: Run Detection ---
run_detection = st.sidebar.button("Run Detection")

# --- Load VAE features ---
with open("models/vae_features.txt") as f:
    vae_features = [line.strip() for line in f if line.strip()]

# --- Load GAN numeric columns ---
gan_ckpt = torch.load("models/ss_gan_dns.pth", weights_only=False)
numeric_cols = gan_ckpt["numeric_cols"]
scaler = StandardScaler()
scaler.mean_ = gan_ckpt["scaler_mean"]
scaler.scale_ = gan_ckpt["scaler_scale"]

# --- Detection Logic ---
def run_hybrid_detection(df, vae_features, numeric_cols, clustering_method, clustering_params):
    # Prepare VAE and GAN inputs
    X_vae = df[vae_features].values
    X_all = scaler.transform(df[numeric_cols])
    input_dim = len(vae_features)
    latent_dim = 8
    vae = VAE(input_dim, latent_dim)
    vae(np.zeros((1, input_dim), dtype=np.float32))
    vae.load_weights("models/vae_final.weights.h5")
    disc = Discriminator(len(numeric_cols))
    disc.load_state_dict(gan_ckpt["disc"])
    disc.eval()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    disc.to(DEVICE)

    # Batch VAE
    vae_recon = vae.predict(X_vae, verbose=0)
    recon_errors = np.mean((vae_recon - X_vae) ** 2, axis=1)
    THR_VAE = 0.015
    THR_GAN = 0.6
    anomalies = []
    original_rows = []
    attack_types = []
    for idx in range(len(X_vae)):
        if recon_errors[idx] < THR_VAE:
            continue
        x_gan_np = X_all[idx]
        x_tensor = torch.tensor(x_gan_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            adv_score, cls_prob = disc(x_tensor)
            adv_prob = torch.sigmoid(adv_score).item()
        if adv_prob < THR_GAN:
            anomalies.append(X_vae[idx])
            original_rows.append(df.iloc[idx])
            attack_types.append(df.iloc[idx].get('label', 'unknown'))
    if not anomalies:
        return None, None, None, None, None
    # Clustering
    X_anom = StandardScaler().fit_transform(anomalies)
    if clustering_method == "KMeans":
        clusterer = KMeans(n_clusters=clustering_params["n_clusters"], random_state=42)
    else:
        clusterer = DBSCAN(eps=clustering_params["eps"], min_samples=clustering_params["min_samples"])
    labels = clusterer.fit_predict(X_anom)
    clustered_df = pd.DataFrame(original_rows)
    clustered_df["cluster"] = labels
    if attack_types:
        clustered_df["attack_type"] = attack_types
    return clustered_df, labels, X_anom, anomalies, attack_types

# --- Main UI Logic ---
if run_detection:
    with st.spinner("Running Hybrid NIDS detection..."):
        clustered_df, labels, X_anom, anomalies, attack_types = run_hybrid_detection(
            df, vae_features, numeric_cols, clustering_method, clustering_params)
    if clustered_df is None:
        st.success("✅ No confirmed anomalies found.")
    else:
        st.success(f"🚨 {len(anomalies)} anomalies confirmed. Clustering...")
        st.write("### Clustered Anomalies Table")
        st.dataframe(clustered_df)
        # Cluster summary
        st.write("### Cluster Summary")
        for cluster_id in sorted(set(labels)):
            count = (labels == cluster_id).sum()
            st.write(f"**Cluster {cluster_id}: {count} samples**")
            cluster_samples = clustered_df[clustered_df["cluster"] == cluster_id]
            st.write(cluster_samples.describe(include='all').T)
        # Attack type summary
        if attack_types:
            st.write("### Attack type counts among detected anomalies:")
            from collections import Counter
            for attack, count in Counter(attack_types).items():
                st.write(f"- {attack}: {count}")
        # 2D Visualization
        st.write("### 2D Cluster Visualization (PCA)")
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_anom)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
        ax.add_artist(legend1)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Anomaly Clusters (PCA)")
        st.pyplot(fig)
        # Download report
        st.write("### Download Clustered Anomalies Report")
        csv = clustered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "clustered_anomalies.csv", "text/csv")
else:
    st.info("Upload/select data and click 'Run Detection' to begin.") 