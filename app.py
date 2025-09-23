import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# ------------------ Load and preprocess data ------------------
df = pd.read_csv("IoT_Intrusion.csv", low_memory=False)

# Update the normal labels based on your dataset
normal_labels = ['Normal', 'Benign']  # Adjust this list as needed
df['label_binary'] = df['label'].apply(lambda x: 0 if x in normal_labels else 1)

# Check the distribution of labels
print("Label distribution before preprocessing:\n", df['label_binary'].value_counts())

# Dynamically select numeric columns as features
features = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Using the following features: {features}")

# Drop rows with NaN values in the selected features
df = df.dropna(subset=features)

# ------------------ Remove Outliers ------------------
# Use IQR to filter outliers for each feature
Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)
IQR = Q3 - Q1

# Define a mask for non-outlier rows
non_outliers = ~((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)

# Comment out the outlier removal step if it removes all normal samples
# df = df[non_outliers]

# Check the distribution after outlier removal
print("Label distribution after outlier removal:\n", df['label_binary'].value_counts())

# Fill any remaining NaN values with the mean of the respective feature
df[features] = df[features].fillna(df[features].mean(numeric_only=True))

# ------------------ Scale and Split Data ------------------
X = df[features].values
y = df['label_binary'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split X_scaled into training (for VAE) and testing (for evaluation)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Extract only normal samples for VAE training
X_train_vae = X_train_full[y_train_full == 0]

# ------------------ VAE Model ------------------
input_dim = X_train_vae.shape[1]
latent_dim = 8

# Custom VAE Loss Layer
class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.input_dim = input_dim

    def call(self, inputs):
        inputs, outputs, z_mean, z_log_var = inputs

        # Reconstruction loss
        reconstruction_loss = tf.reduce_sum(tf.square(inputs - outputs), axis=1)

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)

        # Add the total loss to the model
        self.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))
        return outputs

# Encoder
inputs = Input(shape=(input_dim,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
d = layers.Dense(32, activation='relu')(z)
d = layers.Dense(64, activation='relu')(d)
outputs = layers.Dense(input_dim, activation='sigmoid')(d)

# Add the custom loss layer
vae_outputs = VAELossLayer(input_dim)([inputs, outputs, z_mean, z_log_var])
vae = Model(inputs, vae_outputs)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# ------------------ Train ------------------
print(f"Training VAE on {X_train_vae.shape[0]} normal samples.")
vae.fit(
    X_train_vae, X_train_vae,
    epochs=100,
    batch_size=32,
    shuffle=True,
    verbose=1,
    callbacks=[early_stopping]
)

# ------------------ Prediction ------------------
X_pred = vae.predict(X_test)
recon_errors = np.mean(np.square(X_test - X_pred), axis=1)
recon_errors_normal_test = recon_errors[y_test == 0]

# Threshold tuning
threshold = np.percentile(recon_errors_normal_test, 97)

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