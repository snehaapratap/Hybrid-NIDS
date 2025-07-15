# train_attack_classifier.py (future file)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("data/detected_anomalies_clustered.csv")
X = df.drop(columns=['cluster', 'label'])  # or keep label if you want
y = df['cluster']

clf = RandomForestClassifier()
clf.fit(X, y)

