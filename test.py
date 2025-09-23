import pandas as pd

# Load your dataset
df = pd.read_csv("IoT_Intrusion.csv")

# Choose an attack type present in your data (e.g., "DDoS-UDP_Flood")
attack_type = "DDoS-UDP_Flood"  # Change this to any attack type you want to test

# The 9 features used in manual input (updated to match dataset columns)
manual_features = [
    "flow_duration", "Header_Length", "Duration", "Rate", "Srate", "Drate",
    "ack_flag_number", "syn_flag_number", "fin_flag_number"
]

# Get the first row for the chosen attack type
row = df[df["label"] == attack_type][manual_features].iloc[0]
print(row)

print("Columns in the dataset:", df.columns.tolist())