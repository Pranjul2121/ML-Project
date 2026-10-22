
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

output_file = DATA_DIR / "ai4i2020.csv"

print("Downloading AI4I Predictive Maintenance dataset...")

dataset = fetch_ucirepo(id=601)

features = dataset.data.features
targets = dataset.data.targets

df = pd.concat([features, targets], axis=1)
df.to_csv(output_file, index=False)

print(f"\nDataset saved to: {output_file}")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFailure distribution:")
print(df["Machine failure"].value_counts())
print("\nFirst five rows:")
print(df.head())
