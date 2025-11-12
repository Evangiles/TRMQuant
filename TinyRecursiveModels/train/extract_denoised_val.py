"""
Extract val portion from denoised full dataset.

Uses split_info.json to determine split point dynamically.
"""

import pandas as pd
import json
from pathlib import Path

print("Extracting denoised val set...")

# Load split metadata
split_info_path = Path("TinyRecursiveModels/split_info.json")
if not split_info_path.exists():
    raise FileNotFoundError(
        f"Split metadata not found: {split_info_path}\n"
        "Please run: python TinyRecursiveModels/train/split_train_val.py"
    )

with open(split_info_path, 'r') as f:
    split_info = json.load(f)

n_train = split_info['n_train']
n_val = split_info['n_val']
print(f"Loaded split info: train={n_train}, val={n_val}")

# Load denoised full dataset
df_denoised = pd.read_csv("train_denoised.csv")
print(f"Loaded denoised: {len(df_denoised)} rows")

# Validate
if len(df_denoised) != split_info['total_rows']:
    print(f"⚠️  WARNING: Denoised data has {len(df_denoised)} rows, "
          f"expected {split_info['total_rows']} rows")

# Extract val portion
df_denoised_val = df_denoised.iloc[n_train:].copy()

print(f"Extracted val: {len(df_denoised_val)} rows")

# Save
df_denoised_val.to_csv("val_denoised.csv", index=False)
print("Saved: val_denoised.csv")
