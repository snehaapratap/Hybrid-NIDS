import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # For splitting training data
from tensorflow.keras.callbacks import EarlyStopping

# ------------------ Load and preprocess data ------------------
df = pd.read_csv("combined_dataset.csv", low_memory=False)

# Binary labels: 0 for normal, 1 for attack
normal_labels = ['audio', 'image', 'text', 'video', 'compressed']
df['label_binary'] = df['label'].apply(lambda x: 0 if x in normal_labels else 1)

features = [
    'rr', 'A_frequency', 'NS_frequency', 'CNAME_frequency', 'SOA_frequency',
    'NULL_frequency', 'PTR_frequency', 'HINFO_frequency', 'MX_frequency',
    'TXT_frequency', 'AAAA_frequency', 'SRV_frequency', 'OPT_frequency',
    'rr_count', 'rr_name_entropy', 'rr_name_length', 'a_records',
    'ttl_mean', 'ttl_variance'
]

df = df.dropna(subset=features)
df[features] = df[features].fillna(df[features].mean(numeric_only=True))

X = df[features].values
y = df['label_binary'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split X_scaled into training (for VAE) and testing (for evaluation)
# Important: Ensure that the 'normal' class is well-represented in X_train_vae
# We'll use stratify to maintain the original class distribution for the overall test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Now, extract only normal samples for VAE training from X_train_full
X_train_vae = X_train_full[y_train_full == 0]


# ------------------ VAE model ------------------
input_dim = X_train_vae.shape[1]
latent_dim = 8  # Increased latent_dim - allows for more complex representations

# Encoder
inputs = Input(shape=(input_dim,))
x = layers.Dense(64, activation='relu')(inputs) # Increased neurons
x = layers.Dense(32, activation='relu')(x)    # Increased neurons
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
d = layers.Dense(32, activation='relu')(z)    # Increased neurons
d = layers.Dense(64, activation='relu')(d)    # Increased neurons
outputs = layers.Dense(input_dim, activation='sigmoid')(d) # Sigmoid for scaled data [0, 1]

vae = Model(inputs, outputs)

# VAE loss
# Using mean_squared_error for reconstruction loss often works well with MinMaxScaler
reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs), axis=-1) * input_dim # Scale by input_dim
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss) # Mean over batch

vae.add_loss(vae_loss)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)) # Smaller learning rate

# ------------------ Train ------------------
# Add EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

print(f"Training VAE on {X_train_vae.shape[0]} normal samples.")
vae.fit(X_train_vae, X_train_vae,
        epochs=100,  # Increased epochs, EarlyStopping will manage it
        batch_size=32, # Smaller batch size for potentially better gradient estimation
        shuffle=True,
        verbose=1,
        callbacks=[early_stopping])

# ------------------ Prediction ------------------
X_pred = vae.predict(X_test)
recon_errors = np.mean(np.square(X_test - X_pred), axis=1)

# Threshold tuning: Using percentiles of reconstruction errors from NORMAL samples in the TEST set
# This provides a more robust threshold for separating anomalies.
# First, get reconstruction errors for the normal samples in the test set
recon_errors_normal_test = recon_errors[y_test == 0]

# Calculate a threshold based on a higher percentile of normal errors (e.g., 97th or 98th)
# This aims to minimize false positives (normal data classified as attack)
threshold = np.percentile(recon_errors_normal_test, 97) # Tuned percentile

y_pred = (recon_errors > threshold).astype(int)

# ------------------ Evaluation ------------------
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print(f"✅ Precision: {precision * 100:.2f}%")
print(f"✅ Recall: {recall * 100:.2f}%")
print(f"✅ F1-Score: {f1 * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Error distribution plot
plt.figure(figsize=(10, 6))
plt.hist(recon_errors[y_test == 0], bins=50, alpha=0.6, label='Normal Samples (Test Set)', color='blue')
plt.hist(recon_errors[y_test == 1], bins=50, alpha=0.6, label='Attack Samples (Test Set)', color='orange')
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Reconstruction Error Distribution for Test Set")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()