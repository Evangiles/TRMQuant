"""
Extract val portion from denoised full dataset.
"""

import pandas as pd

print("Extracting denoised val set...")

# Load denoised full dataset
df_denoised = pd.read_csv("train_denoised.csv")
print(f"Loaded denoised: {len(df_denoised)} rows")

# Extract val portion (last 20%)
n_train = 7192
df_denoised_val = df_denoised.iloc[n_train:].copy()

print(f"Extracted val: {len(df_denoised_val)} rows")

# Save
df_denoised_val.to_csv("val_denoised.csv", index=False)
print("Saved: val_denoised.csv")
