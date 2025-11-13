"""
Extract val portion from denoised full dataset.

Uses split_info.json to determine split point dynamically.

Usage:
    python extract_denoised_val.py [denoised_input.csv] [val_output.csv]
"""

import sys
import pandas as pd
import json
from pathlib import Path

# Parse command line arguments
if len(sys.argv) >= 2:
    denoised_input = sys.argv[1]
else:
    denoised_input = "train_denoised.csv"

if len(sys.argv) >= 3:
    val_output = sys.argv[2]
else:
    # Derive output name from input
    val_output = denoised_input.replace("train_denoised", "val_denoised")
    if val_output == denoised_input:
        val_output = "val_denoised.csv"

print(f"Extracting denoised val set...")
print(f"  Input: {denoised_input}")
print(f"  Output: {val_output}")

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
<<<<<<< HEAD
df_denoised = pd.read_csv(denoised_input)
=======
df_denoised = pd.read_csv("train_denoised_v3.csv")
>>>>>>> b490f70 (Add trained denoiser models and leak-free datasets)
print(f"Loaded denoised: {len(df_denoised)} rows")

# Validate
if len(df_denoised) != split_info['total_rows']:
    print(f"⚠️  WARNING: Denoised data has {len(df_denoised)} rows, "
          f"expected {split_info['total_rows']} rows")

# Extract val portion
df_denoised_val = df_denoised.iloc[n_train:].copy()

print(f"Extracted val: {len(df_denoised_val)} rows")

# Save
<<<<<<< HEAD
df_denoised_val.to_csv(val_output, index=False)
print(f"Saved: {val_output}")
=======
df_denoised_val.to_csv("val_denoised_v3.csv", index=False)
print("Saved: val_denoised.csv")
>>>>>>> b490f70 (Add trained denoiser models and leak-free datasets)
