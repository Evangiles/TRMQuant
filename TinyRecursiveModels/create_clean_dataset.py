"""Create clean dataset with NaN threshold filtering and leak-safe imputation.

Strategy:
1. Keep only features with NaN <= 20%
2. Start from row 1006 (where most features become valid)
3. Use forward fill only (no backward fill - no leakage!)
4. Keep D* features despite being binary (low impact expected)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("Creating Clean Dataset with Leak-Safe Strategy")
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
print(f"Original features: {len(feature_cols)}")

# Step 1: Filter features by NaN threshold (20%)
print(f"\n=== Step 1: Filter Features (NaN <= 20%) ===")
nan_threshold = 20.0

filtered_features = []
removed_features = []

for col in feature_cols:
    nan_pct = df[col].isna().sum() / len(df) * 100
    if nan_pct <= nan_threshold:
        filtered_features.append(col)
    else:
        removed_features.append((col, nan_pct))

print(f"Features kept: {len(filtered_features)}")
print(f"Features removed: {len(removed_features)}")

if removed_features:
    print(f"\nRemoved features (NaN > 20%):")
    for feat, pct in sorted(removed_features, key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {pct:.1f}% NaN")

# Check D* features
d_features = [f for f in filtered_features if f.startswith('D')]
print(f"\nD* features kept: {len(d_features)} (binary dummy variables, low impact expected)")

# Step 2: Start from row 1006
print(f"\n=== Step 2: Start from Row 1006 ===")
start_row = 1006
df_clean = df.iloc[start_row:].reset_index(drop=True)
print(f"Original rows: {len(df)}")
print(f"Clean rows: {len(df_clean)} (from row {start_row})")
print(f"Rows removed: {start_row} ({start_row/len(df)*100:.1f}%)")

# Step 3: Check NaN in filtered features after row 1006
print(f"\n=== Step 3: NaN Status After Filtering ===")
X_filtered = df_clean[filtered_features]

nan_counts_after = []
for col in filtered_features:
    nan_count = X_filtered[col].isna().sum()
    nan_pct = nan_count / len(X_filtered) * 100
    if nan_count > 0:
        nan_counts_after.append((col, nan_count, nan_pct))

if nan_counts_after:
    print(f"Features still with NaN after row 1006: {len(nan_counts_after)}")
    for feat, count, pct in sorted(nan_counts_after, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat}: {count} NaN ({pct:.2f}%)")
else:
    print("No NaN remaining after row 1006!")

# Step 4: Forward fill only (leak-safe!)
print(f"\n=== Step 4: Forward Fill (Leak-Safe) ===")
X_clean = X_filtered.copy()
X_clean = X_clean.ffill()  # Forward fill only!

# Check remaining NaN (only possible at first row)
remaining_nan = X_clean.isna().sum().sum()
print(f"NaN after forward fill: {remaining_nan}")

if remaining_nan > 0:
    # Fill first row NaN with column median (calculated on available data)
    print("Filling first row NaN with column median...")
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            col_median = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(col_median)

    final_nan = X_clean.isna().sum().sum()
    print(f"Final NaN count: {final_nan}")

# Step 5: Create final clean dataset
print(f"\n=== Step 5: Create Final Dataset ===")

# Reconstruct with metadata (match original train.csv column order)
# Original order: date_id, features, targets
df_final = pd.concat([
    df_clean[['date_id']].reset_index(drop=True),
    X_clean.reset_index(drop=True),
    df_clean[target_cols].reset_index(drop=True)
], axis=1)

print(f"Final shape: {df_final.shape}")
print(f"Final features: {len(filtered_features)}")
print(f"  Continuous features: {len([f for f in filtered_features if not f.startswith('D')])}")
print(f"  Binary D* features: {len([f for f in filtered_features if f.startswith('D')])}")

# Save
output_path = 'train_clean.csv'
df_final.to_csv(output_path, index=False)
print(f"\n=== Saved to {output_path} ===")

# Summary statistics
print(f"\n=== Summary ===")
print(f"Original dataset:")
print(f"  Rows: {len(df)}, Features: {len(feature_cols)}")
print(f"Clean dataset:")
print(f"  Rows: {len(df_final)} (-{start_row}, -{start_row/len(df)*100:.1f}%)")
print(f"  Features: {len(filtered_features)} (-{len(removed_features)}, -{len(removed_features)/len(feature_cols)*100:.1f}%)")
print(f"  Total data retained: {(len(df_final) * len(filtered_features)) / (len(df) * len(feature_cols)) * 100:.1f}%")

# Save feature list
with open('clean_features.txt', 'w') as f:
    f.write("Clean Features (NaN <= 20%):\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total: {len(filtered_features)} features\n\n")

    # Group by family
    families = {}
    for feat in filtered_features:
        family = feat[0]
        families.setdefault(family, []).append(feat)

    for family in sorted(families.keys()):
        f.write(f"\n{family}* family: {len(families[family])} features\n")
        for feat in families[family]:
            f.write(f"  {feat}\n")

print("\n=== Feature list saved to clean_features.txt ===")

print("\n" + "=" * 80)
print("Clean dataset created successfully!")
print("Next: Train diffusion model with train_clean.csv")
print("=" * 80)
