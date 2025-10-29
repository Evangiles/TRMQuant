"""Analyze NaN patterns in train.csv"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('train.csv')

# Get feature columns
cols = list(df.columns)
date_idx = cols.index('date_id')
target_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
first_target = min(cols.index(c) for c in target_cols)
feature_cols = cols[date_idx + 1:first_target]

print(f"=== NaN Pattern Analysis ===")
print(f"Total rows: {len(df)}")
print(f"Total features: {len(feature_cols)}")
print()

# Analyze each feature
feature_stats = []

for col in feature_cols:
    series = df[col]
    nan_mask = series.isna()
    nan_count = nan_mask.sum()
    total = len(series)

    if nan_count == 0:
        pattern = "No NaN"
        leading_nans = 0
        trailing_nans = 0
        middle_nans = 0
        first_valid = 0
        last_valid = total - 1
    elif nan_count == total:
        pattern = "All NaN"
        leading_nans = total
        trailing_nans = 0
        middle_nans = 0
        first_valid = -1
        last_valid = -1
    else:
        non_nan_idx = np.where(~nan_mask)[0]
        first_valid = non_nan_idx[0]
        last_valid = non_nan_idx[-1]

        leading_nans = first_valid
        trailing_nans = total - last_valid - 1
        middle_nans = nan_count - leading_nans - trailing_nans

        if middle_nans == 0 and leading_nans > 0 and trailing_nans == 0:
            pattern = "Leading only"
        elif middle_nans == 0 and leading_nans == 0 and trailing_nans > 0:
            pattern = "Trailing only"
        else:
            pattern = "Scattered"

    feature_stats.append({
        'feature': col,
        'total_nan': nan_count,
        'nan_pct': nan_count / total * 100,
        'pattern': pattern,
        'leading_nan': leading_nans,
        'trailing_nan': trailing_nans,
        'middle_nan': middle_nans,
        'first_valid_row': first_valid,
        'last_valid_row': last_valid,
        'valid_rows': total - nan_count
    })

# Convert to DataFrame and sort by NaN count
stats_df = pd.DataFrame(feature_stats)
stats_df = stats_df.sort_values('total_nan', ascending=False)

# Print full table
print("=" * 120)
print(f"{'Feature':<10} {'Total NaN':<12} {'NaN %':<10} {'Pattern':<15} {'Leading':<10} {'Trailing':<10} {'Middle':<10} {'Valid Rows':<12}")
print("=" * 120)

for _, row in stats_df.iterrows():
    print(f"{row['feature']:<10} {row['total_nan']:<12} {row['nan_pct']:<10.1f} {row['pattern']:<15} {row['leading_nan']:<10} {row['trailing_nan']:<10} {row['middle_nan']:<10} {row['valid_rows']:<12}")

print("=" * 120)

# Summary statistics
print(f"\n=== Pattern Summary ===")
pattern_counts = stats_df['pattern'].value_counts()
for pattern, count in pattern_counts.items():
    print(f"{pattern}: {count} features")

print(f"\n=== NaN Statistics ===")
print(f"Max leading NaN: {stats_df['leading_nan'].max()}")
print(f"Min leading NaN (excluding 0): {stats_df[stats_df['leading_nan'] > 0]['leading_nan'].min()}")
print(f"Mean leading NaN: {stats_df[stats_df['leading_nan'] > 0]['leading_nan'].mean():.0f}")

# Find global valid range
max_leading = stats_df['leading_nan'].max()
min_trailing = stats_df['trailing_nan'].min()
total_rows = len(df)

print(f"\n=== Global Valid Range ===")
print(f"Valid data starts from row: {max_leading}")
print(f"Valid data ends at row: {total_rows - min_trailing}")
print(f"Total valid rows: {total_rows - max_leading - min_trailing} ({(total_rows - max_leading - min_trailing) / total_rows * 100:.1f}%)")

# Save to CSV
stats_df.to_csv('nan_analysis.csv', index=False)
print(f"\n=== Results saved to nan_analysis.csv ===")
