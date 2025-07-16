#!/usr/bin/env python
# coding: utf-8
"""
Converted from Untitled3.ipynb
+ Added code to save the final trained VAE model as a PyTorch
  checkpoint called `vae_final.pt`.
"""

# ------------------ Imports ------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns  # For better confusion‑matrix visualization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import ops  # Keras 3 compatible ops

# Ensure TensorFlow uses GPU if available
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
if tf.config.list_physical_devices("GPU"):
    print("TensorFlow is using GPU.")
else:
    print(
        "TensorFlow is running on CPU. "
        "Consider installing TF‑GPU support for faster training."
    )

# ------------------ Load and preprocess data ------------------
# Make sure 'IoT_Intrusion.csv' is in the same directory,
# or provide the full path to the file.
df = pd.read_csv("data/IoT_Intrusion.csv", low_memory=False)

# Features selected for the VAE
features = [
    "flow_duration",
    "Header_Length",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
]

# Handle missing values
df = df.dropna(subset=features)  # Drop rows with NaNs in selected features
df[features] = df[features].fillna(
    df[features].mean(numeric_only=True)
)  # Fill remaining NaNs

# Scale features to [0, 1]
X = df[features].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]  # Feature count
latent_dim = 8  # Size of latent space

# ------------------ VAE Definition ------------------
class VAE(Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_dense_1 = layers.Dense(64, activation="relu")
        self.encoder_dense_2 = layers.Dense(32, activation="relu")
        self.z_mean_layer = layers.Dense(latent_dim)
        self.z_log_var_layer = layers.Dense(latent_dim)

        # Decoder
        self.decoder_dense_1 = layers.Dense(32, activation="relu")
        self.decoder_dense_2 = layers.Dense(64, activation="relu")
        self.decoder_output = layers.Dense(input_dim, activation="sigmoid")

    # Encode input to latent mean and log‑var
    def encode(self, inputs):
        x = self.encoder_dense_1(inputs)
        x = self.encoder_dense_2(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        return z_mean, z_log_var

    # Reparameterization trick
    def sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    # Decode latent vector
    def decode(self, z):
        d = self.decoder_dense_1(z)
        d = self.decoder_dense_2(d)
        return self.decoder_output(d)

    # Forward pass
    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.sampling(z_mean, z_log_var)
        return self.decode(z)

    # Custom training step with VAE loss
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]  # Only take the input part (X)
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(data)
            z = self.sampling(z_mean, z_log_var)
            reconstruction = self.decode(z)

            # Reconstruction loss
            recon_loss = ops.mean(ops.square(data - reconstruction), axis=-1) * self.input_dim
            # KL‑divergence loss
            kl_loss = -0.5 * ops.sum(
                1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1
            )
            total_loss = ops.mean(recon_loss + kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": ops.mean(recon_loss),
            "kl_loss": ops.mean(kl_loss),
        }


# ------------------ Generate Simulated Ground Truth ------------------
print("\n--- Generating Simulated Ground‑Truth Labels ---")
vae_temp = VAE(input_dim, latent_dim)
vae_temp.compile(optimizer=tf.keras.optimizers.Adam(0.001))
vae_temp.fit(X_scaled, X_scaled, epochs=15, batch_size=32, verbose=0)

recon_errors_full = np.mean(np.square(X_scaled - vae_temp.predict(X_scaled)), axis=1)

anomaly_percentile = 90  # Use 90th percentile as anomaly cutoff
threshold_fake_true = np.percentile(recon_errors_full, anomaly_percentile)
y_fake_true = (recon_errors_full > threshold_fake_true).astype(int)
print(f"Simulated‑label threshold: {threshold_fake_true:.4f}")

# ------------------ Cross‑Validation ------------------
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_cm, all_precision, all_recall, all_f1, all_accuracy = [], [], [], [], []

print(f"\n--- {n_splits}-Fold Cross‑Validation ---")
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_fake_true), 1):
    print(f"\n--- Fold {fold}/{n_splits} ---")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_fake_true[train_idx], y_fake_true[test_idx]

    vae_fold = VAE(input_dim, latent_dim)
    vae_fold.compile(optimizer=tf.keras.optimizers.Adam(0.001))
    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    vae_fold.fit(X_train, X_train, epochs=100, batch_size=32, verbose=0, callbacks=[es])

    X_pred = vae_fold.predict(X_test)
    recon_errors = np.mean(np.square(X_test - X_pred), axis=1)
    threshold_fold = np.percentile(recon_errors, anomaly_percentile)

    y_pred = (recon_errors > threshold_fold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    all_cm.append(cm)
    all_precision.append(precision_score(y_test, y_pred, zero_division=0))
    all_recall.append(recall_score(y_test, y_pred, zero_division=0))
    all_f1.append(f1_score(y_test, y_pred, zero_division=0))
    all_accuracy.append(accuracy_score(y_test, y_pred))

# ------------------ Train on Full Data ------------------
print("\n--- Training Final Model on Full Dataset ---")
vae_final = VAE(input_dim, latent_dim)
vae_final.compile(optimizer=tf.keras.optimizers.Adam(0.001))
early_stopping_final = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
vae_final.fit(
    X_scaled,
    X_scaled,
    epochs=100,
    batch_size=32,
    shuffle=True,
    verbose=0,
    callbacks=[early_stopping_final],
)

# ---- Save the final trained model to .pt (PyTorch checkpoint) ----
# We first try to export the weights as a PyTorch state_dict so that they can be
# re‑used directly in a PyTorch environment. If PyTorch isn't installed (or any
# other problem occurs), we fall back to saving in Keras format.
try:
    import torch
    from collections import OrderedDict

    state_dict = OrderedDict()
    for layer in vae_final.layers:
        for idx, w in enumerate(layer.get_weights()):
            state_dict[f"{layer.name}_{idx}"] = torch.tensor(w)
    torch.save(state_dict, "vae_final.pt")
    print("\nSaved VAE weights to 'vae_final.pt' (PyTorch).")
except Exception as e:
    print(f"\n[Warning] Could not save to '.pt' → {e}")
    print("Saving model in Keras format instead → 'vae_final.keras'")
    vae_final.save("vae_final.keras")

# ------------------ Evaluate on Full Data ------------------
X_pred_final = vae_final.predict(X_scaled)
recon_errors_final = np.mean(np.square(X_scaled - X_pred_final), axis=1)

overall_threshold = threshold_fake_true  # Same percentile used earlier
y_pred_anomaly_final = (recon_errors_final > overall_threshold).astype(int)

print(f"\nOverall Anomaly Threshold: {overall_threshold:.4f}")
print(
    f"Overall Number of detected anomalies: {np.sum(y_pred_anomaly_final)} "
    f"({np.mean(y_pred_anomaly_final)*100:.2f}%)"
)

# Plot reconstruction‑error distribution
plt.figure(figsize=(12, 7))
sns.histplot(
    recon_errors_final[y_fake_true == 0],
    bins=50,
    label="True Normal (Simulated)",
    kde=True,
    stat="density",
    alpha=0.6,
)
sns.histplot(
    recon_errors_final[y_fake_true == 1],
    bins=50,
    label="True Anomaly (Simulated)",
    kde=True,
    stat="density",
    alpha=0.6,
)
plt.axvline(
    overall_threshold,
    color="red",
    linestyle="--",
    label=f"Overall Threshold ({overall_threshold:.4f})",
)
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Reconstruction Error Distribution with Simulated Labels")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# Confusion matrix for entire data
cm_overall = confusion_matrix(y_fake_true, y_pred_anomaly_final)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_overall,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Normal", "Anomaly"],
    yticklabels=["Normal", "Anomaly"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Overall Confusion Matrix (Simulated Labels)")
plt.show()

print(f"Overall Precision: {precision_score(y_fake_true, y_pred_anomaly_final, zero_division=0):.4f}")
print(f"Overall Recall:    {recall_score(y_fake_true, y_pred_anomaly_final, zero_division=0):.4f}")
print(f"Overall F1-Score:  {f1_score(y_fake_true, y_pred_anomaly_final, zero_division=0):.4f}")
print(f"Overall Accuracy:  {accuracy_score(y_fake_true, y_pred_anomaly_final):.4f}")
