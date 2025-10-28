#  Hybrid-NIDS

**A Multi-Layered Hybrid Network Intrusion Detection System using VAE, GAN, and HDBSCAN for Detecting Known and Unknown Threats**


##  Overview

Network anomalies refer to behavior that is either hidden or unexpected, which can cause **cyberattacks, performance issues, or security breaches**, making early detection critical to maintaining network stability.

This research introduces a **multi-layered Network Intrusion Detection System (NIDS)** that leverages a **Variational Autoencoder (VAE)**, a **Generative Adversarial Network (GAN)**, and a **clustering algorithm (HDBSCAN)** to detect and classify network anomalies efficiently.

Traditional signature-based systems often fail to detect **unknown or evolving threats**, as they rely on pre-defined attack patterns. In contrast, the proposed **Hybrid-NIDS** focuses on identifying **unusual or hidden deviations** from normal network behavior—making it effective against both **known and zero-day attacks**.

The system operates in three key stages:

1. **Anomaly Identification (VAE):** The autoencoder captures and reconstructs normal traffic patterns, identifying deviations that differ from expected network behavior.
2. **Verification (GAN):** The GAN discriminator validates these deviations using its deeper understanding of normal and abnormal patterns, effectively reducing false positives.
3. **Classification (HDBSCAN):** Verified anomalies are clustered based on similarity, enabling interpretable and analysis-ready categorization for security analysts.

This **multi-layered hybrid approach** improves detection accuracy, reduces manual analysis effort, and provides a **scalable and reliable defense mechanism** against modern cybersecurity threats.


## ⚙️ Methodology

###  Variational Autoencoder (VAE) – Anomaly Detection

* The VAE is trained on **benign traffic** to learn the statistical distribution of normal behavior.
* When malicious traffic is encountered, the model produces a **higher reconstruction error**, enabling early anomaly detection.

###  Generative Adversarial Network (GAN) – Anomaly Verification

* The GAN discriminator validates the anomalies identified by the VAE.
* By leveraging adversarial training, the GAN differentiates between normal and anomalous traffic with improved precision.
* This reduces **false positives**, confirming only genuine threats for further analysis.

### HDBSCAN – Attack Clustering and Classification

* The verified anomalies are then passed to **HDBSCAN (Hierarchical Density-Based Spatial Clustering)**.
* HDBSCAN groups the anomalies into clusters based on similarity in their latent features.
* This unsupervised clustering step enables analysts to understand attack patterns and emerging threats without manual labeling.


##  Experimental Setup

* **Dataset:** Preprocessed IoT intrusion dataset (`IoT_Intrusion.csv`).
* **Environment:** Python 3.8+, trained models using GPU acceleration for faster convergence.
* **Models Used:**

  * *VAE:* Encoder-decoder architecture with latent dimension compression for anomaly detection.
  * *GAN:* Generator-discriminator network trained on VAE outputs for refined attack understanding.
  * *HDBSCAN:* Unsupervised clustering to categorize detected attacks into interpretable groups.
* **Results:**

  * Effective separation between benign and malicious data in latent space.
  * Reduced false alarms compared to standalone models.
  * Robust detection of zero-day and evolving network attacks.


##  Quick Start

All preprocessing and model training are already complete.
You can directly launch the interactive detection interface.

### **Setup**

```bash
git clone https://github.com/snehaapratap/Hybrid-NIDS.git
cd Hybrid-NIDS
pip install -r requirements.txt
```

### **Run the Streamlit App**

```bash
streamlit run streamlit_detect.py
```

This interface allows users to upload test data, view detected anomalies, and explore attack clusters visually.


## Results & Observations

* **VAE** effectively isolates anomalies from normal network traffic.
* **GAN** validates and refines anomalies, minimizing false detections.
* **HDBSCAN** groups the confirmed anomalies into distinct clusters representing different attack categories.
* The hybrid model demonstrates **high adaptability** to previously unseen attacks.

Visualizations such as `real_data_clusters.png` and `generated_samples_clusters.png` show clear cluster separations, reinforcing the model’s interpretability and precision.



##  Future Scope

* Integrate real-time traffic capture for online intrusion detection.
* Extend hybrid learning to federated environments for privacy-preserving IDS.
* Explore interpretability frameworks for explainable intrusion detection.
* Benchmark performance on larger and multi-source datasets.


##  Contributors

* **Sneha Pratap** — [@snehaapratap](https://github.com/snehaapratap)
* **Shreya Channalli** — [@Shreya-Channalli](https://github.com/Shreya-Channalli)
* **Chaitra Devadig** — [@ChaitraDevadig03](https://github.com/ChaitraDevadig03)


