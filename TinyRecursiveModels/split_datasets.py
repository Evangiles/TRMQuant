"""Split datasets into train/val folders for leak-free processing.

Naming convention:
- origin.csv, clean.csv, clean2.csv (original full datasets)
- train/origin_train.csv, train/clean_train.csv, ... (80% train split)
- val/origin_val.csv, val/clean_val.csv, ... (20% val split)
"""

import pandas as pd
import os

print("=" * 80)
print("Splitting Datasets into Train/Val Folders")
print("=" * 80)

# Create folders
os.makedirs('train', exist_ok=True)
os.makedirs('val', exist_ok=True)

# Define datasets to split
datasets = {
    'origin': 'train.csv',
    'clean': 'train_clean.csv',
    'clean2': 'train_clean_v2.csv',
}

split_ratio = 0.8

print("\nProcessing datasets...")
for name, filepath in datasets.items():
    if not os.path.exists(filepath):
        print(f"  Skipping {name}: {filepath} not found")
        continue

    print(f"\n  {name}: {filepath}")
    df = pd.read_csv(filepath)

    # Split
    split_idx = int(len(df) * split_ratio)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    # Save
    train_path = f'train/{name}_train.csv'
    val_path = f'val/{name}_val.csv'

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(f"    Train: {train_path} ({len(df_train)} rows)")
    print(f"    Val:   {val_path} ({len(df_val)} rows)")

print("\n" + "=" * 80)
print("Folder Structure:")
print("=" * 80)
print("train/")
print("  - origin_train.csv (80% of original data)")
print("  - clean_train.csv (80% of NaN-filtered data)")
print("  - clean2_train.csv (80% of all-features data)")
print("")
print("val/")
print("  - origin_val.csv (20% of original data)")
print("  - clean_val.csv (20% of NaN-filtered data)")
print("  - clean2_val.csv (20% of all-features data)")

print("\n" + "=" * 80)
print("Next Steps:")
print("=" * 80)
print("1. Train diffusion model:")
print("   uv run train_diffusion_denoiser.py --path train/clean_train.csv")
print("")
print("2. Denoise train:")
print("   uv run denoise_csv.py --input train/clean_train.csv --output train/clean_denoised_train.csv")
print("")
print("3. Denoise val:")
print("   uv run denoise_csv.py --input val/clean_val.csv --output val/clean_denoised_val.csv")
print("")
print("4. Test with LightGBM (leak-free!)")
print("=" * 80)
