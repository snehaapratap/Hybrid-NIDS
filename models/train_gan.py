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

# Added imports for clustering and visualization
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CSV_PATH = "IoT_Intrusion.csv"
NORMAL_LABELS = [
    "audio", "image", "text", "video", "compressed", "exe",
    # add / remove as you decide what is benign
]
LATENT_DIM = 100
BATCH = 256
EPOCHS = 20
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
    loss_D = torch.tensor(0.0, device=DEVICE)
    loss_G = torch.tensor(0.0, device=DEVICE)
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
    # Move print statement here, using last batch loss values
    print(f"Epoch {epoch:02d}/{EPOCHS} "
          f"| D: {loss_D.item():.3f}  G: {loss_G.item():.3f}")

# --------------------------------------------------------------------
# 5)  Save weights
# --------------------------------------------------------------------
torch.save(G.state_dict(), "models/gan_generator.pt")
torch.save(D.state_dict(), "models/gan_discriminator.pt")
torch.save({
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "numeric_cols": numeric_cols,
}, "models/gan_scaler.pth")

print("✅  Semi‑Supervised GAN trained and saved to models/gan_generator.pt, models/gan_discriminator.pt, and models/gan_scaler.pth")

# --------------------------------------------------------------------
# 6)  HDBScan clustering on generated and real data features
# --------------------------------------------------------------------
G.eval()
with torch.no_grad():
    # Generate synthetic samples
    n_samples = 1000
    z = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
    generated_samples = G(z).cpu().numpy()

    # Real data features (scaled)
    real_features = scaler.transform(df[numeric_cols])

    # Cluster generated samples with optimizations
    print("Starting generated samples clustering...")
    
    # Apply PCA for dimensionality reduction if dimensions are high
    if generated_samples.shape[1] > 50:
        print("Applying PCA for dimensionality reduction on generated samples...")
        pca_gen = PCA(n_components=50)
        generated_samples_reduced = pca_gen.fit_transform(generated_samples)
    else:
        generated_samples_reduced = generated_samples
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,  # Increased for better performance
        min_samples=5,        # Increased for better performance
        core_dist_n_jobs=1
    )
    gen_labels = clusterer.fit_predict(generated_samples_reduced)

    # Debug print original labels
    print(f"Original generated sample cluster labels: {np.unique(gen_labels)}")

    # Relabel clusters to start from 0, replace -1 for outliers with 2
    unique_labels = np.unique(gen_labels)
    if -1 in unique_labels:
        unique_labels = unique_labels[unique_labels != -1]
    # Instead of mapping, subtract min label from all labels except -1
    min_label = unique_labels.min() if unique_labels.size > 0 else 0
    gen_labels_relabel = np.array([label - min_label if label != -1 else 2 for label in gen_labels])

    # Debug print relabeled clusters
    print(f"Relabeled generated sample cluster labels: {np.unique(gen_labels_relabel)}")

    print("HDBScan clustering on generated samples:")
    print(f"Number of clusters found: {len(set(gen_labels_relabel)) - (1 if -1 in gen_labels_relabel else 0)}")
    print(f"Cluster labels distribution: {np.bincount(gen_labels_relabel[gen_labels_relabel >= 0])}")

    # Save generated samples with cluster labels to CSV
    import pandas as pd
    gen_df = pd.DataFrame(generated_samples, columns=numeric_cols)
    gen_df['cluster_label'] = gen_labels_relabel
    gen_df.to_csv("generated_samples_clusters.csv", index=False)
    print("Generated samples with cluster labels saved to generated_samples_clusters.csv")

    # Cluster real data features with performance optimizations
    print("Starting real data clustering with optimizations...")
    
    # Sample the data to reduce computational load
    sample_size = min(5000, len(real_features))  # Limit to 5000 samples max
    if len(real_features) > sample_size:
        print(f"Sampling {sample_size} samples from {len(real_features)} total samples for clustering...")
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(real_features), sample_size, replace=False)
        real_features_sample = real_features[sample_indices]
    else:
        print("Using full dataset for clustering...")
        real_features_sample = real_features
    
    # Apply PCA for dimensionality reduction
    print("Applying PCA for dimensionality reduction...")
    pca_real = PCA(n_components=min(50, real_features_sample.shape[1]))  # Reduce to 50 dimensions max
    real_features_reduced = pca_real.fit_transform(real_features_sample)
    print(f"Reduced dimensions from {real_features_sample.shape[1]} to {real_features_reduced.shape[1]}")
    
    # Use optimized HDBSCAN parameters for better performance
    real_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,  # Increased from 2 for better performance
        min_samples=5,        # Increased from 1 for better performance
        cluster_selection_epsilon=0.5,
        core_dist_n_jobs=1   # Limit parallel jobs to prevent memory issues
    )
    
    print("Performing HDBSCAN clustering on real data...")
    real_labels_sample = real_clusterer.fit_predict(real_features_reduced)
    
    # Map labels back to original indices
    if len(real_features) > sample_size:
        real_labels = np.full(len(real_features), -1)
        real_labels[sample_indices] = real_labels_sample
    else:
        real_labels = real_labels_sample

    # Debug print original labels
    print(f"Original real data cluster labels: {np.unique(real_labels)}")

    # Relabel clusters to start from 0, keep -1 for outliers
    unique_labels_real = np.unique(real_labels)
    if -1 in unique_labels_real:
        unique_labels_real = unique_labels_real[unique_labels_real != -1]
    # Instead of mapping, subtract min label from all labels except -1
    min_label_real = unique_labels_real.min() if unique_labels_real.size > 0 else 0
    real_labels_relabel = np.array([label - min_label_real if label != -1 else -1 for label in real_labels])

    # Debug print relabeled clusters
    print(f"Relabeled real data cluster labels: {np.unique(real_labels_relabel)}")

    print("HDBScan clustering on real data features:")
    print(f"Number of clusters found: {len(set(real_labels_relabel)) - (1 if -1 in real_labels_relabel else 0)}")
    print(f"Cluster labels distribution: {np.bincount(real_labels_relabel[real_labels_relabel >= 0])}")

    # Save real data features with cluster labels to CSV
    real_df = df[numeric_cols].copy()
    real_df['cluster_label'] = real_labels_relabel
    real_df.to_csv("real_data_clusters.csv", index=False)
    print("Real data features with cluster labels saved to real_data_clusters.csv")

    # Visualization of clustering results using PCA
    pca = PCA(n_components=2)

    # Plot generated samples clusters
    gen_pca = pca.fit_transform(generated_samples)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(gen_pca[:, 0], gen_pca[:, 1], c=gen_labels, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title('PCA of Generated Samples Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('generated_samples_clusters.png')
    plt.close()

    # Plot real data clusters
    real_pca = pca.fit_transform(real_features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(real_pca[:, 0], real_pca[:, 1], c=real_labels, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title('PCA of Real Data Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('real_data_clusters.png')
    plt.close()

    print("Cluster visualizations saved as 'generated_samples_clusters.png' and 'real_data_clusters.png'")
