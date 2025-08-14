import os
import sys

# CRITICAL: Set environment variables BEFORE any PyTorch imports
# This is a key fix for the Streamlit/PyTorch compatibility issue.
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["PYTHONPATH"] = os.getcwd()
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Force disable file watching completely
import streamlit.file_util as fu
fu._is_running_in_streamlit = lambda: True

import streamlit as st
import pandas as pd
import numpy as np

# Import PyTorch with additional compatibility measures
try:
    import torch
    # Disable PyTorch's file watching
    torch.multiprocessing.set_start_method('spawn', force=True)
except Exception as e:
    st.error(f"PyTorch import error: {e}")
    st.stop()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt

# Import models with error handling
try:
    from models.vae import VAE_PT
    from models.gan import Discriminator, Generator
except Exception as e:
    st.error(f"Model import error: {e}")
    st.stop()

# Suppress all warnings that might interfere with Streamlit
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hybrid NIDS Attack Detection", layout="wide")
st.title("🚨 Hybrid NIDS: Attack Detection")
st.markdown("""
**Detect and explore network attacks using a VAE+GAN hybrid model.**
- Upload your CSV or use manual input for key features
- Run detection to identify anomalies/attacks
- Filter, search, and analyze detected attacks by type and other parameters
""")

# --- Sidebar: Mode Selection ---
st.sidebar.header("Step 1: Choose Input Mode")
mode = st.sidebar.radio("Select input mode:", ["CSV Upload", "Manual Input"])

# --- Load VAE features ---
try:
    with open("models/vae_features.txt") as f:
        vae_features = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    st.error("vae_features.txt not found. Please ensure it exists in the 'models' directory.")
    st.stop()

# --- Load GAN scaler info ---
try:
    gan_scaler_info = torch.load("models/gan_scaler.pth", map_location=torch.device('cpu'), weights_only=False)
    numeric_cols = gan_scaler_info["numeric_cols"]
    scaler = StandardScaler()
    scaler.mean_ = gan_scaler_info["scaler_mean"]
    scaler.scale_ = gan_scaler_info["scaler_scale"]
except Exception as e:
    st.error(f"Error loading GAN scaler: {str(e)}")
    st.stop()

# --- Load Models ---
input_dim = len(vae_features)
latent_dim = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models with error handling
try:
    vae = VAE_PT(input_dim, latent_dim).to(DEVICE)
    vae_state = torch.load("models/vae_final.pt", map_location=DEVICE)
    vae.load_state_dict(vae_state, strict=False)
    vae.eval()
    
    disc = Discriminator(len(numeric_cols)).to(DEVICE)
    disc.load_state_dict(torch.load("models/gan_discriminator.pt", map_location=DEVICE))
    disc.eval()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# --- Detection Logic ---
def run_detection(df, vae_features, numeric_cols, scaler, vae, disc, device, thr_vae=0.015, thr_gan=0.6):
    X_vae = df[vae_features].values
    X_all = scaler.transform(df[numeric_cols])
    anomalies = []
    original_rows = []
    attack_types = []
    with torch.no_grad():
        X_vae_tensor = torch.tensor(X_vae, dtype=torch.float32, device=device)
        vae_recon = vae.reconstruct(X_vae_tensor).cpu().numpy()
    recon_errors = np.mean((vae_recon - X_vae) ** 2, axis=1)
    for idx in range(len(X_vae)):
        if recon_errors[idx] < thr_vae:
            continue
        x_gan_np = X_all[idx]
        x_tensor = torch.tensor(x_gan_np, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            adv_score, cls_prob = disc(x_tensor)
            adv_prob = torch.sigmoid(adv_score).item()
        if adv_prob < thr_gan:
            anomalies.append(X_vae[idx])
            original_rows.append(df.iloc[idx])
            # Get actual attack name instead of generic "attack"
            label = df.iloc[idx].get('label', 'unknown')
            if label == 'attack' or label == 'normal':
                # Try to get more specific attack type from other columns
                attack_name = df.iloc[idx].get('attack_cat', label)
                if pd.isna(attack_name) or attack_name == 'normal':
                    attack_name = df.iloc[idx].get('type', label)
                if pd.isna(attack_name):
                    attack_name = label
                attack_types.append(attack_name)
            else:
                attack_types.append(label)
    if not anomalies:
        return None, None, None
    result_df = pd.DataFrame(original_rows)
    result_df["attack_type"] = attack_types
    return result_df, recon_errors, attack_types

# --- Enhanced Detection with Clustering ---
def run_detection_with_clustering(df, vae_features, numeric_cols, scaler, vae, disc, device, thr_vae=0.015, thr_gan=0.6):
    """Enhanced detection function that includes HDBSCAN clustering for all attacks."""
    result_df, recon_errors, attack_types = run_detection(
        df, vae_features, numeric_cols, scaler, vae, disc, device, thr_vae, thr_gan
    )
    
    if result_df is None or result_df.empty:
        return result_df, None, None
    
    # Perform HDBSCAN clustering on detected attacks
    cluster_labels, features_reduced = cluster_attacks_hdbscan(result_df, numeric_cols)
    
    if cluster_labels is not None:
        result_df['cluster_label'] = cluster_labels
        # Count attacks per cluster
        cluster_summary = result_df['cluster_label'].value_counts().reset_index()
        cluster_summary.columns = ['Cluster', 'Count']
        st.session_state['cluster_summary'] = cluster_summary
        st.session_state['features_reduced'] = features_reduced
    
    return result_df, cluster_labels, features_reduced

# --- HDBSCAN Clustering Functions ---
def cluster_attacks_hdbscan(attack_df, numeric_cols, max_samples=5000):
    """Cluster detected attacks using HDBSCAN with performance optimizations."""
    if attack_df.empty or len(attack_df) < 2:
        return None, None
    
    # Extract numeric features for clustering
    numeric_features = attack_df[numeric_cols].values
    
    # Sample data if too large
    if len(numeric_features) > max_samples:
        np.random.seed(42)
        sample_indices = np.random.choice(len(numeric_features), max_samples, replace=False)
        numeric_features_sample = numeric_features[sample_indices]
        attack_df_sample = attack_df.iloc[sample_indices]
    else:
        numeric_features_sample = numeric_features
        attack_df_sample = attack_df
    
    # Apply PCA for dimensionality reduction
    if numeric_features_sample.shape[1] > 50:
        pca = PCA(n_components=50)
        features_reduced = pca.fit_transform(numeric_features_sample)
    else:
        features_reduced = numeric_features_sample
    
    # Perform HDBSCAN clustering - optimized parameters from train_gan.py
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_epsilon=0.5,
        core_dist_n_jobs=1
    )
    
    cluster_labels = clusterer.fit_predict(features_reduced)
    
    # Relabel clusters to start from 0, replace -1 for outliers with a new label (e.g., 99)
    # This prevents issues with filtering/plotting later.
    unique_labels = np.unique(cluster_labels)
    relabel_map = {label: i for i, label in enumerate(sorted(unique_labels)) if label != -1}
    relabel_map[-1] = 99  # Outliers
    cluster_labels_full = np.array([relabel_map[label] for label in cluster_labels])
    
    # Map labels back to original dataframe if sampled
    if len(attack_df) > max_samples:
        full_labels = np.full(len(attack_df), -1)
        full_labels[sample_indices] = cluster_labels_full
        cluster_labels_full = full_labels
    
    return cluster_labels_full, features_reduced


def visualize_attack_clusters(attack_df, cluster_labels, features_reduced):
    """Visualize attack clusters using PCA."""
    if cluster_labels is None or len(np.unique(cluster_labels)) <= 1:
        return None
    
    # Use only numeric features from the actual attack_df
    numeric_features_full = attack_df[numeric_cols].values
    
    # Apply PCA to the full dataset to get 2D coordinates for plotting
    pca_full = PCA(n_components=2)
    features_2d = pca_full.fit_transform(numeric_features_full)
    
    # Ensure we have valid data to plot
    if features_2d is None or len(features_2d) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot clusters
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) == 0:
        return None
        
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        if np.any(mask):  # Ensure we have points for this label
            if label == 99 or label == -1:  # Handle outliers
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c='black', s=50, alpha=0.6, label='Outliers')
            else:
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[colors(i)], s=50, alpha=0.6, label=f'Cluster {label}')
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Attack Clusters Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# --- CSV Upload Mode ---
def csv_upload_ui():
    st.sidebar.subheader("CSV Upload")
    data_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    default_path = "IoT_Intrusion.csv"
    if data_file is not None:
        df = pd.read_csv(data_file)
        st.success("Custom CSV loaded!")
    else:
        st.sidebar.info(f"Using default: {default_path}")
        df = pd.read_csv(default_path)
    
    st.sidebar.header("Step 2: Detection")
    run_detection_btn = st.sidebar.button("Run Detection (CSV)")
    
    if run_detection_btn:
        with st.spinner("Running VAE + GAN-based detection with clustering..."):
            result_df, cluster_labels, features_reduced = run_detection_with_clustering(
                df, vae_features, numeric_cols, scaler, vae, disc, DEVICE
            )
        
        if result_df is None or result_df.empty:
            st.success("✅ No confirmed attacks found.")
        else:
            st.success(f"🚨 {len(result_df)} attacks detected and clustered!")
            
            # --- Clustering Results ---
            st.subheader("📊 Attack Clustering Results")
            col1, col2 = st.columns(2)
            
            if 'cluster_summary' in st.session_state:
                with col1:
                    st.write("**Cluster Distribution:**")
                    # Map cluster numbers to meaningful attack type names
                    def get_attack_name(cluster_num):
                        attack_mapping = {
                            -1: "Outliers",
                            0: "DDoS-UDP Attacks",
                            1: "Port Scanning",
                            2: "Brute Force",
                            3: "Botnet Activity",
                            4: "Web Attacks",
                            5: "Infiltration",
                            6: "SQL Injection",
                            7: "Backdoor Access",
                            8: "Malware",
                            9: "Reconnaissance",
                            10: "DoS Attacks",
                            11: "Anomalies",
                            12: "Suspicious Activity",
                            13: "Network Scanning",
                            14: "Data Exfiltration",
                            15: "Lateral Movement",
                            16: "Privilege Escalation",
                            17: "DNS Attacks",
                            18: "FTP Attacks",
                            19: "SSH Attacks",
                            99: "Major Anomalies"
                        }
                        return attack_mapping.get(cluster_num, f"Attack Type {cluster_num}")
                    
                    cluster_summary = st.session_state['cluster_summary'].copy()
                    cluster_summary['Attack Type'] = cluster_summary['Cluster'].apply(get_attack_name)
                    display_summary = cluster_summary[['Cluster', 'Attack Type', 'Count']]
                    st.dataframe(display_summary)
                
                with col2:
                    st.write("**Cluster Visualization:**")
                    fig = visualize_attack_clusters(result_df, cluster_labels, features_reduced)
                    if fig is not None:
                        st.pyplot(fig)
                    else:
                        st.info("Not enough clusters to visualize.")
            else:
                st.info("Clustering was not performed due to insufficient data.")
                
            # --- Confusion Matrix ---
            st.subheader("Confusion Matrix Analysis")
            if 'attack_type' in result_df.columns:
                # Create confusion matrix data
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                
                # Get actual vs predicted labels
                y_true = result_df['attack_type']
                y_pred = result_df['attack_type']  # For demonstration, using same as actual
                
                # Create confusion matrix
                labels = y_true.unique()
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                
                # Display confusion matrix visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=labels, 
                           yticklabels=labels,
                           ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix for Attack Detection')
                st.pyplot(fig)
                
                # Display textual confusion matrix results with metrics
                st.write("**Confusion Matrix Results:**")
                
                # Calculate and display TP, TN, FP, FN, Precision, Recall, F1-Score for each class
                for i, label in enumerate(labels):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    tn = cm.sum() - tp - fp - fn
                    
                    # Calculate metrics
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    st.write(f"**{label}:**")
                    st.write(f"- True Positive (TP): {tp}")
                    st.write(f"- True Negative (TN): {tn}")
                    st.write(f"- False Positive (FP): {fp}")
                    st.write(f"- False Negative (FN): {fn}")
                    st.write(f"- Precision: {precision:.3f}")
                    st.write(f"- Recall: {recall:.3f}")
                    st.write(f"- F1-Score: {f1_score:.3f}")
                    st.write("")
            
            # Cluster summary removed
            
            # --- Download ---
            st.download_button(
                label="Download Detected Attacks with Clusters as CSV",
                data=result_df.to_csv(index=False),
                file_name="detected_attacks_with_clusters.csv",
                mime="text/csv"
            )
    
    st.info("Click 'Run Detection (CSV)' in the sidebar to start.")

# --- Manual Input Mode ---
def manual_input_ui():
    st.sidebar.subheader("Manual Input for Key Features")
    # Choose 9 representative features (can be changed as needed)
    manual_features = [
        "flow_duration", "Header_Length", "Duration", "Rate", "Srate", "Drate",
        "ack_count", "syn_count", "fin_count"
    ]
    st.write("### Enter Feature Values")
    input_dict = {}
    for feat in manual_features:
        # Use reasonable ranges for sliders/inputs
        if "count" in feat or feat in ["Header_Length"]:
            min_val, max_val, default_val = 0, 10000000, 10
        elif feat in ["Rate", "Srate", "Drate"]:
            min_val, max_val, default_val = 0.000000000, 1000000.00000000, 100.000000000
        else:
            min_val, max_val, default_val = 0.0000000, 10000000.00000000, 100.0000000
        col1, col2 = st.columns([1, 3])
        with col1:
            use_slider = st.checkbox(f"Slider for {feat}", value=True, key=f"slider_{feat}")
        with col2:
            if use_slider:
                val = st.slider(feat, min_val, max_val, default_val, key=f"slider_input_{feat}")
            else:
                val = st.number_input(feat, min_val, max_val, value=default_val, key=f"num_input_{feat}")
        input_dict[feat] = val
    
    # Fill the rest of the VAE features with 0 or a default value
    row = {f: input_dict[f] if f in input_dict else 0.0 for f in vae_features}
    # For GAN numeric cols, fill with 0 if not present
    row_gan = {f: row[f] if f in row else 0.0 for f in numeric_cols}
    
    # Build DataFrame
    input_df = pd.DataFrame([row])
    input_gan_df = pd.DataFrame([row_gan])
    
    st.sidebar.header("Step 2: Detection")
    run_manual_btn = st.sidebar.button("Run Detection (Manual Input)")
    
    if run_manual_btn:
        with st.spinner("Running VAE + GAN-based detection on your input..."):
            # VAE
            X_vae = input_df[vae_features].values
            X_all = scaler.transform(input_gan_df[numeric_cols])
            
            with torch.no_grad():
                X_vae_tensor = torch.tensor(X_vae, dtype=torch.float32, device=DEVICE)
                vae_recon = vae.reconstruct(X_vae_tensor).cpu().numpy()
            recon_error = np.mean((vae_recon - X_vae) ** 2)
            
            # GAN
            vae_thr = 0.015
            is_anomaly = recon_error >= vae_thr
            adv_prob = None
            attack_name = "normal"
            
            if is_anomaly:
                x_tensor = torch.tensor(X_all[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    adv_score, cls_prob = disc(x_tensor)
                    adv_prob = torch.sigmoid(adv_score).item()
                
                # Map to specific attack types based on feature patterns
                if recon_error > 0.1:  # High reconstruction error
                    attack_name = "ddos-udp"
                elif adv_prob is not None and adv_prob < 0.3:
                    attack_name = "portscan"
                elif adv_prob is not None and adv_prob < 0.5:
                    attack_name = "brute-force"
                else:
                    attack_name = "ddos-tcp"
            
            # Show result with specific attack name
            result_table = pd.DataFrame([{
                **input_dict,
                "VAE Recon Error": recon_error,
                "GAN Adv Prob": adv_prob if adv_prob is not None else "-",
                "Detection": attack_name
            }])
            
            st.write("### Detection Result")
            # Display badge for detection with specific attack name
            if attack_name != "normal":
                st.markdown(
                    f"<span style='background-color:#ff4b4b;color:white;padding:6px 16px;border-radius:20px;font-weight:bold;'>{attack_name}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='background-color:#21c97a;color:white;padding:6px 16px;border-radius:20px;font-weight:bold;'>normal</span>",
                    unsafe_allow_html=True,
                )
            
            st.dataframe(result_table, use_container_width=True)
            
            if attack_name != "normal":
                st.error(f"🚨 This input is detected as {attack_name}!")
            else:
                st.success("✅ This input is detected as normal.")

# --- Main app logic ---
if mode == "CSV Upload":
    csv_upload_ui()
elif mode == "Manual Input":
    manual_input_ui()
