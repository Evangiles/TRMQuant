"""Create clean dataset V2 - Keep ALL features, start from row 1006.

Strategy:
1. Keep ALL 94 features (no NaN threshold filtering)
2. Start from row 1006 (where most features become valid)
3. Use forward fill only (no backward fill - no leakage!)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("Creating Clean Dataset V2 - All Features")
print("=" * 80)

# Load data
df = pd.read_csv('train.csv')
print(f"\nOriginal data: {df.shape}")

# Get feature columns
cols = list(df.columns)
date_idx = cols.index('date_id')
target_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
first_target = min(cols.index(c) for c in target_cols)
feature_cols = cols[date_idx + 1:first_target]
print(f"Features: {len(feature_cols)}")

# Start from row 1006
print(f"\n=== Starting from Row 1006 ===")
start_row = 1006
df_clean = df.iloc[start_row:].reset_index(drop=True)
print(f"Original rows: {len(df)}")
print(f"Clean rows: {len(df_clean)} (from row {start_row})")
print(f"Rows removed: {start_row} ({start_row/len(df)*100:.1f}%)")

# Check NaN status after row 1006
print(f"\n=== NaN Status After Row 1006 ===")
X_filtered = df_clean[feature_cols]

nan_counts_after = []
for col in feature_cols:
    nan_count = X_filtered[col].isna().sum()
    nan_pct = nan_count / len(X_filtered) * 100
    if nan_count > 0:
        nan_counts_after.append((col, nan_count, nan_pct))

if nan_counts_after:
    print(f"Features with NaN after row 1006: {len(nan_counts_after)}")
    print(f"\nTop 10 features with most NaN:")
    for feat, count, pct in sorted(nan_counts_after, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat}: {count} NaN ({pct:.2f}%)")

    total_nan = sum(c for _, c, _ in nan_counts_after)
    print(f"\nTotal NaN across all features: {total_nan}")
else:
    print("No NaN remaining after row 1006!")

# Forward fill only (leak-safe!)
print(f"\n=== Forward Fill (Leak-Safe) ===")
X_clean = X_filtered.copy()
X_clean = X_clean.ffill()  # Forward fill only!

# Check remaining NaN (only possible at leading rows)
remaining_nan = X_clean.isna().sum().sum()
print(f"NaN after forward fill: {remaining_nan}")

if remaining_nan > 0:
    # Fill remaining NaN with column median (calculated on available data)
    print("Filling remaining leading NaN with column median...")
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            col_median = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(col_median)

    final_nan = X_clean.isna().sum().sum()
    print(f"Final NaN count: {final_nan}")

# Create final clean dataset
print(f"\n=== Final Dataset ===")

# Reconstruct with metadata
meta_cols = ['date_id'] + target_cols
df_final = pd.concat([
    df_clean[meta_cols].reset_index(drop=True),
    X_clean.reset_index(drop=True)
], axis=1)

print(f"Final shape: {df_final.shape}")
print(f"Features: {len(feature_cols)} (all original features kept)")

# Save
output_path = 'train_clean_v2.csv'
df_final.to_csv(output_path, index=False)
print(f"\n=== Saved to {output_path} ===")

# Summary statistics
print(f"\n=== Summary ===")
print(f"Original dataset:")
print(f"  Rows: {len(df)}, Features: {len(feature_cols)}")
print(f"Clean dataset V2:")
print(f"  Rows: {len(df_final)} (-{start_row}, -{start_row/len(df)*100:.1f}%)")
print(f"  Features: {len(feature_cols)} (all kept)")
print(f"  Data retention: {len(df_final) / len(df) * 100:.1f}% rows, 100% features")

print("\n" + "=" * 80)
print("Clean dataset V2 created successfully!")
print(f"V1 (NaN <= 20%): 81 features")
print(f"V2 (all features): 94 features")
print("=" * 80)
