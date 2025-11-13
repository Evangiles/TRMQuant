"""
Split train.csv into train (80%) and val (20%) for proper evaluation.

This ensures denoising model is trained only on train set,
and evaluation is done only on unseen val set.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def split_train_val(input_csv: str, train_ratio: float = 0.8):
    """
    Split dataset into train and val sets.

    Args:
        input_csv: Path to train.csv
        train_ratio: Ratio of training data (default: 0.8)
    """
    print("="*80)
    print("SPLIT TRAIN/VAL FOR LEAK-FREE EVALUATION")
    print("="*80)

    # Load data
    print(f"\nLoading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}")

    # Calculate split point
    n_train = int(len(df) * train_ratio)
    n_val = len(df) - n_train

    print(f"\nSplitting {train_ratio:.0%} / {1-train_ratio:.0%}:")
    print(f"  Train: {n_train} rows")
    print(f"  Val:   {n_val} rows")

    # Split (chronological order for time series)
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:].copy()

    # Save
    output_dir = Path(input_csv).parent

    train_path = output_dir / "train_only.csv"
    val_path = output_dir / "val_only.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(f"\nSaved:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Train denoising models on train_only.csv:")
    print("   bash TinyRecursiveModels/train/train_all_clusters.sh")
    print("   (Modify script to use train_only.csv)")
    print("\n2. Denoise full dataset (train + val):")
    print("   python inference/denoise_dataset.py --input_csv train.csv --output_csv train_denoised.csv")
    print("\n3. Extract denoised val set:")
    print("   Will create val_denoised.csv automatically")
    print("\n4. Evaluate on val set only:")
    print("   python evaluation/validate_denoising.py --original val_only.csv --denoised val_denoised.csv")

    # Save split metadata for reproducibility
    import json
    from datetime import datetime

    split_info = {
        'n_train': n_train,
        'n_val': n_val,
        'train_ratio': train_ratio,
        'total_rows': len(df),
        'split_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_path = output_dir / "split_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"  Metadata: {metadata_path}")
    print(f"\nVal set indices: {n_train} to {len(df)}")

    return n_train, n_val


if __name__ == "__main__":
    split_train_val("TinyRecursiveModels/train.csv", train_ratio=0.8)
