#!/usr/bin/env python
# coding: utf-8

# In[6]:

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns # For better confusion matrix visualization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import ops # Import ops for Keras 3 compatible operations

# ------------------ Load and preprocess data ------------------
df = pd.read_csv("data/stateless_features-light_image.pcap.csv", low_memory=False)

features = [
    'FQDN_count', 'subdomain_length', 'upper', 'lower', 'numeric',
    'entropy', 'special', 'labels', 'labels_max', 'labels_average',
    'len', 'subdomain'
]

df = df.dropna(subset=features)
df[features] = df[features].fillna(df[features].mean(numeric_only=True))

X = df[features].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]
latent_dim = 8

# ------------------ VAE model (Subclassed) ------------------
class VAE(Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_dense_1 = layers.Dense(64, activation='relu')
        self.encoder_dense_2 = layers.Dense(32, activation='relu')
        self.z_mean_layer = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var_layer = layers.Dense(latent_dim, name='z_log_var')

        self.decoder_dense_1 = layers.Dense(32, activation='relu')
        self.decoder_dense_2 = layers.Dense(64, activation='relu')
        self.decoder_output = layers.Dense(input_dim, activation='sigmoid')

    def encode(self, inputs):
        x = self.encoder_dense_1(inputs)
        x = self.encoder_dense_2(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        return z_mean, z_log_var

    def decode(self, z):
        d = self.decoder_dense_1(z)
        d = self.decoder_dense_2(d)
        outputs = self.decoder_output(d)
        return outputs

    def sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.sampling(z_mean, z_log_var)
        outputs = self.decode(z)
        return outputs

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(data)
            z = self.sampling(z_mean, z_log_var)
            reconstruction = self.decode(z)

            reconstruction_loss = ops.mean(ops.square(data - reconstruction), axis=-1) * self.input_dim
            kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
            total_loss = ops.mean(reconstruction_loss + kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "reconstruction_loss": ops.mean(reconstruction_loss), "kl_loss": ops.mean(kl_loss)}

# ------------------ Generate Simulated Ground Truth (y_fake_true) ------------------
# Train a VAE once on the entire dataset to establish a baseline for reconstruction errors
# to create our simulated ground truth.
print("\n--- Generating Simulated Ground Truth Labels ---")
vae_temp = VAE(input_dim, latent_dim)
vae_temp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# Use a small number of epochs for this initial training for speed
vae_temp.fit(X_scaled, X_scaled, epochs=20, batch_size=32, verbose=0)

# Calculate reconstruction errors for the entire dataset
recon_errors_full = np.mean(np.square(X_scaled - vae_temp.predict(X_scaled)), axis=1)

# Define a percentile for anomalies to create pseudo-labels
# Lowering this percentile typically increases recall but may decrease precision.
anomaly_percentile = 90 # Changed from 97 to 90 to aim for higher recall
threshold_fake_true = np.percentile(recon_errors_full, anomaly_percentile)
y_fake_true = (recon_errors_full > threshold_fake_true).astype(int)

print(f"Simulated Anomaly Threshold: {threshold_fake_true:.4f}")
print(f"Number of Simulated Anomalies: {np.sum(y_fake_true)}")
print(f"Percentage of Simulated Anomalies: {np.sum(y_fake_true) / len(y_fake_true) * 100:.2f}%")

# ------------------ 10-Fold Cross-Validation ------------------
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_cm = []
all_precision = []
all_recall = []
all_f1 = []
all_accuracy = []

print(f"\n--- Starting {n_splits}-Fold Cross-Validation ---")

for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y_fake_true)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_true_fold, y_test_fold = y_fake_true[train_index], y_fake_true[test_index] # Pseudo labels for this fold

    # Re-instantiate and compile VAE for each fold to ensure fresh weights
    vae_fold = VAE(input_dim, latent_dim)
    vae_fold.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    early_stopping_fold = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)

    print(f"Training VAE on {X_train_fold.shape[0]} samples (unsupervised) for Fold {fold+1}.")
    vae_fold.fit(X_train_fold, X_train_fold,
                 epochs=100,
                 batch_size=32,
                 shuffle=True,
                 verbose=0, # Set to 0 to suppress per-epoch output during CV
                 callbacks=[early_stopping_fold])

    # Calculate reconstruction errors for training data of this fold (to determine threshold)
    recon_errors_train_fold = np.mean(np.square(X_train_fold - vae_fold.predict(X_train_fold)), axis=1)
    # Calculate reconstruction errors for test data of this fold
    recon_errors_test_fold = np.mean(np.square(X_test_fold - vae_fold.predict(X_test_fold)), axis=1)

    # Determine threshold based on the training data of the current fold
    threshold_fold = np.percentile(recon_errors_train_fold, anomaly_percentile)

    # Classify test data based on the threshold
    y_pred_fold = (recon_errors_test_fold > threshold_fold).astype(int)

    # ------------------ Confusion Matrix & Metrics for current fold ------------------
    cm_fold = confusion_matrix(y_test_fold, y_pred_fold)
    precision_fold = precision_score(y_test_fold, y_pred_fold, zero_division=0)
    recall_fold = recall_score(y_test_fold, y_pred_fold, zero_division=0)
    f1_fold = f1_score(y_test_fold, y_pred_fold, zero_division=0)
    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)

    all_cm.append(cm_fold)
    all_precision.append(precision_fold)
    all_recall.append(recall_fold)
    all_f1.append(f1_fold)
    all_accuracy.append(accuracy_fold)

    print(f"Fold {fold+1} Metrics:")
    print(f"  Threshold: {threshold_fold:.4f}")
    print(f"  Confusion Matrix:\n{cm_fold}")
    print(f"  Precision: {precision_fold:.4f}")
    print(f"  Recall:    {recall_fold:.4f}")
    print(f"  F1-Score:  {f1_fold:.4f}")
    print(f"  Accuracy:  {accuracy_fold:.4f}")

# ------------------ Aggregate and Report Cross-Validation Results ------------------
print(f"\n--- Cross-Validation Results Summary (over {n_splits} folds) ---")
print(f"Average Precision: {np.mean(all_precision):.4f} (+/- {np.std(all_precision):.4f})")
print(f"Average Recall:    {np.mean(all_recall):.4f} (+/- {np.std(all_recall):.4f})")
print(f"Average F1-Score:  {np.mean(all_f1):.4f} (+/- {np.std(all_f1):.4f})")
print(f"Average Accuracy:  {np.mean(all_accuracy):.4f} (+/- {np.std(all_accuracy):.4f})")

# Calculate average confusion matrix
avg_cm = np.mean(all_cm, axis=0).astype(int)
print("\nAverage Confusion Matrix across folds:")
print(avg_cm)

# Plotting the average Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Normal', 'Predicted Anomaly'],
            yticklabels=['True Normal', 'True Anomaly'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Average Confusion Matrix (Simulated Labels)')
plt.show()

# ------------------ Re-run original analysis for plotting reconstruction errors with simulated truth ------------------
# This section is kept for visualizing the overall distribution with the globally determined threshold
print("\n--- Overall Anomaly Detection Results (using global threshold) ---")

# Train VAE on the entire dataset one last time for overall visualization
vae_final = VAE(input_dim, latent_dim)
vae_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
early_stopping_final = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
vae_final.fit(X_scaled, X_scaled,
              epochs=100,
              batch_size=32,
              shuffle=True,
              verbose=0,
              callbacks=[early_stopping_final])

X_pred_final = vae_final.predict(X_scaled)
recon_errors_final = np.mean(np.square(X_scaled - X_pred_final), axis=1)

# The threshold for the overall plot comes from the initial simulated ground truth calculation
overall_threshold = threshold_fake_true
y_pred_anomaly_final = (recon_errors_final > overall_threshold).astype(int)

print(f"\nOverall Anomaly Threshold: {overall_threshold:.4f}")
print(f"Overall Number of detected anomalies: {np.sum(y_pred_anomaly_final)}")
print(f"Overall Percentage of detected anomalies: {np.sum(y_pred_anomaly_final) / len(y_pred_anomaly_final) * 100:.2f}%")

# Plotting the distribution of reconstruction errors with simulated truth overlay
plt.figure(figsize=(12, 7))
sns.histplot(recon_errors_final[y_fake_true == 0], bins=50, color='skyblue', label='True Normal (Simulated)', kde=True, stat='density', alpha=0.6)
sns.histplot(recon_errors_final[y_fake_true == 1], bins=50, color='salmon', label='True Anomaly (Simulated)', kde=True, stat='density', alpha=0.6)
plt.axvline(overall_threshold, color='red', linestyle='--', label=f'Overall Anomaly Threshold ({overall_threshold:.4f})')
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Reconstruction Error Distribution with Simulated Ground Truth")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Display confusion matrix for the overall results using the simulated ground truth
print("\n--- Overall Confusion Matrix (Simulated Labels) ---")
overall_cm = confusion_matrix(y_fake_true, y_pred_anomaly_final)
print(overall_cm)

plt.figure(figsize=(8, 6))
sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Normal', 'Predicted Anomaly'],
            yticklabels=['True Normal', 'True Anomaly'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Overall Confusion Matrix (Simulated Labels)')
plt.show()

print(f"Overall Precision: {precision_score(y_fake_true, y_pred_anomaly_final, zero_division=0):.4f}")
print(f"Overall Recall:    {recall_score(y_fake_true, y_pred_anomaly_final, zero_division=0):.4f}")
print(f"Overall F1-Score:  {f1_score(y_fake_true, y_pred_anomaly_final, zero_division=0):.4f}")
print(f"Overall Accuracy:  {accuracy_score(y_fake_true, y_pred_anomaly_final):.4f}")


# In[ ]:

os.makedirs("models", exist_ok=True)

# Save full model (recommended for Keras 3)
vae_final.save("models/vae_final.keras")

# Save only weights with correct filename format
vae_final.save_weights("models/vae_final.weights.h5")

print("✅ Model successfully saved in 'models/' folder:")
print("- Full model: models/vae_final.keras")
print("- Weights only: models/vae_final.weights.h5")




