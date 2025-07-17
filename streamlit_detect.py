import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from models.vae import VAE_PT
from models.gan import Discriminator

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
with open("models/vae_features.txt") as f:
    vae_features = [line.strip() for line in f if line.strip()]

# --- Load GAN scaler info ---
gan_scaler = torch.load("models/gan_scaler.pth", weights_only=False)
numeric_cols = gan_scaler["numeric_cols"]
scaler = StandardScaler()
scaler.mean_ = gan_scaler["scaler_mean"]
scaler.scale_ = gan_scaler["scaler_scale"]

# --- Load Models ---
input_dim = len(vae_features)
latent_dim = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
vae = VAE_PT(input_dim, latent_dim).to(DEVICE)
vae_state = torch.load("models/vae_final.pt", map_location=DEVICE)
vae.load_state_dict(vae_state, strict=False)
vae.eval()
disc = Discriminator(len(numeric_cols)).to(DEVICE)
disc.load_state_dict(torch.load("models/gan_discriminator.pt", map_location=DEVICE))
disc.eval()

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
            attack_types.append(df.iloc[idx].get('label', 'unknown'))
    if not anomalies:
        return None, None, None
    result_df = pd.DataFrame(original_rows)
    result_df["attack_type"] = attack_types
    return result_df, recon_errors, attack_types

# --- CSV Upload Mode ---
def csv_upload_ui():
    st.sidebar.subheader("CSV Upload")
    data_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    default_path = "data/IoT_Intrusion.csv"
    if data_file is not None:
        df = pd.read_csv(data_file)
        st.success("Custom CSV loaded!")
    else:
        st.sidebar.info(f"Using default: {default_path}")
        df = pd.read_csv(default_path)
    st.sidebar.header("Step 2: Detection")
    run_detection_btn = st.sidebar.button("Run Detection (CSV)")
    if run_detection_btn:
        with st.spinner("Running VAE + GAN-based detection..."):
            result_df, recon_errors, attack_types = run_detection(
                df, vae_features, numeric_cols, scaler, vae, disc, DEVICE
            )
        if result_df is None or result_df.empty:
            st.success("✅ No confirmed attacks found.")
        else:
            st.success(f"🚨 {len(result_df)} attacks detected!")
            # --- Filtering UI ---
            st.subheader("Detected Attacks Table")
            # Dynamic filter by attack type
            attack_types_unique = sorted(result_df["attack_type"].unique())
            selected_types = st.multiselect(
                "Filter by attack type:", attack_types_unique, default=attack_types_unique
            )
            filtered_df = result_df[result_df["attack_type"].isin(selected_types)]
            # Dynamic filter by other columns
            filter_col = st.selectbox(
                "Filter by another column:", [c for c in result_df.columns if c not in ["attack_type"]]
            )
            filter_val = st.text_input(f"Show rows where {filter_col} contains:")
            if filter_val:
                filtered_df = filtered_df[filtered_df[filter_col].astype(str).str.contains(filter_val, case=False)]
            st.dataframe(filtered_df, use_container_width=True)
            # --- Summary ---
            st.subheader("Attack Type Summary")
            summary = filtered_df["attack_type"].value_counts().reset_index()
            summary.columns = ["Attack Type", "Count"]
            st.table(summary)
            st.bar_chart(summary.set_index("Attack Type"))
            # --- Download ---
            st.download_button(
                label="Download Detected Attacks as CSV",
                data=filtered_df.to_csv(index=False),
                file_name="detected_attacks.csv",
                mime="text/csv"
            )
    else:
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
            attack_type = "Normal"
            if is_anomaly:
                x_tensor = torch.tensor(X_all[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    adv_score, cls_prob = disc(x_tensor)
                    adv_prob = torch.sigmoid(adv_score).item()
                # If VAE error is extremely high, override GAN
                if recon_error > (vae_thr * 10):
                    attack_type = "Attack"
                elif adv_prob is not None and adv_prob < 0.6:
                    attack_type = "Attack"
            # Show result
            result_table = pd.DataFrame([{
                **input_dict,
                "VAE Recon Error": recon_error,
                "GAN Adv Prob": adv_prob if adv_prob is not None else "-",
                "Detection": attack_type
            }])
            st.write("### Detection Result")
            # Display badge for detection
            if attack_type == "Attack":
                st.markdown(
                    f"<span style='background-color:#ff4b4b;color:white;padding:6px 16px;border-radius:20px;font-weight:bold;'>Attack</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='background-color:#21c97a;color:white;padding:6px 16px;border-radius:20px;font-weight:bold;'>Normal</span>",
                    unsafe_allow_html=True,
                )
            st.dataframe(result_table, use_container_width=True)
            if attack_type == "Attack":
                st.error("🚨 This input is detected as an ATTACK!")
            else:
                st.success("✅ This input is detected as NORMAL.")
    else:
        st.info("Set feature values and click 'Run Detection (Manual Input)' in the sidebar.")

# --- Main UI Switch ---
if mode == "CSV Upload":
    csv_upload_ui()
else:
    manual_input_ui() 