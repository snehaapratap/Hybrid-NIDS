import pandas as pd
import os

# Define file paths and labels
data_dir = "data/"  # Change if your CSVs are in a different folder
files = {
    "audio": "stateful_features-light_audio.pcap.csv",
    "compressed": "stateful_features-light_compressed.pcap.csv",
    "exe": "stateful_features-light_exe.pcap.csv",
    "image": "stateful_features-light_image.pcap.csv",
    "text": "stateful_features-light_text.pcap.csv",
    "video": "stateful_features-light_video.pcap.csv",
    "stateless_audio": "stateless_features-light_audio.pcap.csv",
    "stateless_compressed": "stateless_features-light_compressed.pcap.csv",
    "stateless_exe": "stateless_features-light_exe.pcap.csv",
    "stateless_image": "stateless_features-light_image.pcap.csv",
}

# Load and label each file
dataframes = []
for label, filename in files.items():
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    df['label'] = label
    dataframes.append(df)

# Combine all into one DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Optional: Save combined file
combined_df.to_csv("combined_dataset_raw.csv", index=False)

print("Combined dataset shape:", combined_df.shape)
