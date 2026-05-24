from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "swell"
    / "archive"
    / "hrv dataset"
    / "data"
    / "final"
    / "train.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "swell"
    / "archive"
    / "hrv dataset"
    / "data"
    / "final"
    / "test.csv"
)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\n=== SWELL DATASET ===\n")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nКолонки:")
print(list(train_df.columns))

print("\nПерші 5 рядків:")
print(train_df.head())

print("\nУнікальні stress conditions:")
print(train_df["condition"].unique())

print("\nРозподіл класів:")
print(train_df["condition"].value_counts())

print("\nHRV features:")

hrv_features = [
    "RMSSD",
    "SDRR",
    "LF_HF",
    "pNN50",
    "MEAN_RR"
]

for feature in hrv_features:
    if feature in train_df.columns:
        print(f"✔ {feature}")