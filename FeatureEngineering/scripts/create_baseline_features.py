"""
Create baseline features with preprocessing only (no feature engineering).
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import preprocess_pipeline


def main():
    print("="*60)
    print("Creating Baseline Features")
    print("="*60)

    # Run preprocessing pipeline
    df, metadata = preprocess_pipeline(
        input_path="data/raw/train.csv",
        output_path="data/processed/train_baseline.csv",
        config_path="configs/preprocessing.yaml"
    )

    print("\n" + "="*60)
    print("Baseline features created successfully!")
    print("="*60)
    print(f"\nOutput: data/processed/train_baseline.csv")
    print(f"Features: {metadata['n_features']}")
    print(f"Train samples: {metadata['train_samples']}")
    print(f"Val samples: {metadata['val_samples']}")


if __name__ == "__main__":
    main()
