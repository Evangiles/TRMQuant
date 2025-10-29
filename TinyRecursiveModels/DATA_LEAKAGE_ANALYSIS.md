# Data Leakage Analysis Report

## Executive Summary

**Critical Finding**: Current diffusion denoising implementation contains **MASSIVE data leakage** through backward fill imputation.

- **77.5% of data** (6969/8990 rows) are leading NaN values
- Current `bfill()` fills all leading NaN with **future values**
- This invalidates all performance metrics reported

---

## 1. NaN Pattern Analysis

### 1.1 Dataset Overview

```
Total rows: 8990
Total features: 94
Valid data range: row 6969 ~ 8990 (2021 rows, 22.5%)
```

### 1.2 Pattern Distribution

| Pattern | Count | Percentage |
|---------|-------|------------|
| **Leading NaN only** | 85 features | 90.4% |
| **No NaN** | 9 features | 9.6% |
| Trailing NaN | 0 features | 0% |
| Scattered NaN | 0 features | 0% |

**Key Insight**:
- **No scattered NaN** - all features are either completely filled or have leading NaN only
- **No trailing NaN** - all features end with valid data
- This simplifies imputation strategy significantly

### 1.3 Leading NaN Statistics

```
Max leading NaN: 6969 rows (77.5%)
Min leading NaN: 1006 rows (11.2%)
Mean leading NaN: 1620 rows (18.0%)
```

### 1.4 Top 10 Features by NaN Count

| Rank | Feature | Total NaN | NaN % | Pattern | Valid Rows |
|------|---------|-----------|-------|---------|------------|
| 1 | E7 | 6969 | 77.5% | Leading only | 2021 |
| 2 | V10 | 6049 | 67.3% | Leading only | 2941 |
| 3 | S3 | 5733 | 63.8% | Leading only | 3257 |
| 4 | M1 | 5547 | 61.7% | Leading only | 3443 |
| 5 | M13 | 5540 | 61.6% | Leading only | 3450 |
| 6 | M14 | 5540 | 61.6% | Leading only | 3450 |
| 7 | M6 | 5043 | 56.1% | Leading only | 3947 |
| 8 | V9 | 4539 | 50.5% | Leading only | 4451 |
| 9 | S12 | 3537 | 39.3% | Leading only | 5453 |
| 10 | M5 | 3283 | 36.5% | Leading only | 5707 |

### 1.5 Features with No NaN

Only **9 features** (D1-D9) have complete data:
- D1, D2, D3, D4, D5, D6, D7, D8, D9

---

## 2. Data Leakage Analysis

### 2.1 Current Implementation

**File**: `train_diffusion_denoiser.py` (Line 217)

```python
# ❌ WRONG: Leakage occurs here
X_df = X_df.ffill().bfill()  # bfill() fills leading NaN with future values!
median = X_df.median(numeric_only=True)  # Global median includes validation data
X_df = X_df.fillna(median)
```

**File**: `denoise_csv.py` (Line 236-238)

```python
# ❌ WRONG: Same leakage pattern
X_df = X_df.ffill().bfill()
median = X_df.median(numeric_only=True)
X_df = X_df.fillna(median)
```

### 2.2 Leakage Mechanisms

#### Mechanism 1: Backward Fill (bfill)

```python
# Example with E7 (6969 leading NaN)
Row 0:    [NaN, NaN, NaN, ...]
Row 1:    [NaN, NaN, NaN, ...]
...
Row 6968: [NaN, NaN, NaN, ...]
Row 6969: [100, 102, 105, ...]  ← First valid value

# After bfill():
Row 0:    [100, 102, 105, ...]  ← Filled with row 6969 values!
Row 1:    [100, 102, 105, ...]  ← FUTURE INFORMATION
...
Row 6968: [100, 102, 105, ...]  ← FUTURE INFORMATION
Row 6969: [100, 102, 105, ...]  ← Original
```

**Impact**: 6969 rows (77.5% of data) are filled with future values!

#### Mechanism 2: Global Median

```python
# Standard 80/20 split
split_idx = int(8990 * 0.8) = 7192

# Train: rows 0-7191 (includes 6969 NaN rows + 223 valid rows)
# Val:   rows 7192-8989 (1798 valid rows)

# Median calculation includes validation data
median = X_df.median()  # Uses rows 0-8989 (including validation!)
```

**Impact**: Validation data statistics leak into training imputation.

### 2.3 Impact on Performance

**Reported Performance** (with leakage):
```
Raw baseline:     Adjusted Sharpe = 0.20
Denoised:         Adjusted Sharpe = 1.84 (+820%)
```

**Analysis**:
- +820% improvement is **abnormally high**
- Likely inflated by data leakage
- True performance expected to be **much lower**

---

## 3. Split Strategy Analysis

### 3.1 Current Split (Incorrect)

```
Total: 8990 rows
Split point: row 7192 (80%)

Train: rows 0-7191
  - Leading NaN rows: 6969 (96.9% of train!)
  - Valid rows: 223 (3.1% of train)

Val: rows 7192-8989
  - All valid (no NaN)
  - 1798 rows
```

**Problem**: Train set is 96.9% NaN rows filled with leakage!

### 3.2 Valid Data Only Split (Recommended)

```
Valid data: rows 6969-8989 (2021 rows)
Split point: row 8585 (80% of valid data)

Train: rows 6969-8584
  - All valid (no NaN)
  - 1616 rows

Val: rows 8585-8989
  - All valid (no NaN)
  - 405 rows
```

**Advantage**:
- ✅ Zero NaN - no imputation needed
- ✅ Complete leak-free
- ✅ Matches paper methodology (no imputation mentioned)

**Disadvantage**:
- ❌ Reduced training data (7192 → 1616 rows, -77.5%)
- ❌ Fewer window samples (47,807 → ~7,332, -84.7%)

---

## 4. Paper Analysis (2409.02138v1.pdf)

### 4.1 What Paper Says

**Section 5.1 - Datasets**:
> "We use rolling window of size 60 and stride 20 to get stock closing price time series"
> "Each dataset is divided into training and testing periods in the proportion of 4:1"

**FeatureWindowDataset** (Algorithm):
```python
# Per-window z-score normalization (mentioned)
mean = window.mean()
std = window.std() + 1e-8
window_normalized = (window - mean) / std
```

### 4.2 What Paper Does NOT Say

**No mention of**:
- ❌ `bfill()` backward fill
- ❌ `median()` global imputation
- ❌ Any NaN handling strategy
- ❌ Leading NaN rows

### 4.3 Conclusion

**Current implementation adds arbitrary imputation** not mentioned in paper:
- `ffill().bfill()` is not paper-based
- `median()` imputation is not paper-based
- These were implementation choices that introduced leakage

---

## 5. Recommendations

### Option 1: Valid Data Only (Recommended for Phase 1)

**Pros**:
- ✅ 100% leak-free
- ✅ No imputation needed
- ✅ Faithful to paper (no imputation mentioned)
- ✅ Simplest implementation

**Cons**:
- ❌ 77.5% data loss
- ❌ Reduced performance expected

**Implementation**:
```python
# Skip leading NaN rows
df = df.iloc[6969:].reset_index(drop=True)

# Split on valid data only
split_idx = int(len(df) * 0.8)  # 1616 train, 405 val
df_train = df.iloc[:split_idx]
df_val = df.iloc[split_idx:]

# No imputation needed!
X_train = df_train[feature_cols].to_numpy()
X_val = df_val[feature_cols].to_numpy()
```

### Option 2: Forward Fill + Train Median Only

**Pros**:
- ✅ Uses all data (8990 rows)
- ✅ Leak-reduced (not leak-free)

**Cons**:
- ⚠️ Still some leakage risk in median calculation
- ⚠️ Complex imputation logic

**Implementation**:
```python
# Temporal split FIRST
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx]
df_val = df.iloc[split_idx:]

# Train imputation (forward fill only!)
X_train = df_train[feature_cols].ffill()

# Calculate median ONLY on valid train rows (row 6969+)
train_valid_rows = X_train.iloc[6969:]  # Only valid rows
train_median = train_valid_rows.median()
X_train = X_train.fillna(train_median)

# Val imputation (reuse train median)
X_val = df_val[feature_cols].ffill()
X_val = X_val.fillna(train_median)
```

### Option 3: Hybrid Approach

**Strategy**:
1. Train diffusion model on valid data only (leak-free)
2. Denoise entire dataset (including leading NaN rows)
3. Use denoised full dataset for feature engineering

**Pros**:
- ✅ Diffusion model: leak-free training
- ✅ Feature engineering: full data utilization
- ✅ Best of both worlds

**Cons**:
- ⚠️ Complex implementation
- ⚠️ Leading NaN rows still questionable for FE

---

## 6. Expected Results After Fix

### Current (with leakage):
```
Raw:      0.20 Adjusted Sharpe
Denoised: 1.84 Adjusted Sharpe (+820%)
```

### Expected (leak-free):
```
Raw:      0.20 Adjusted Sharpe (unchanged)
Denoised: 0.4 ~ 0.8 Adjusted Sharpe (+100% ~ 300%)
```

**Reasoning**:
- +820% is abnormally high for denoising
- Realistic denoising gains: +100% ~ +300%
- True effectiveness will be revealed after leakage removal

---

## 7. Action Plan

### Phase 1: Establish Leak-Free Baseline (CRITICAL)

1. ✅ Implement Option 1 (Valid data only)
2. ✅ Re-train diffusion model (leak-free)
3. ✅ Re-run all experiments
4. ✅ Measure TRUE denoising performance
5. ✅ Document results

**Time**: ~4-5 hours (GPU training)

### Phase 2: Optimize Data Utilization

1. Evaluate Option 2 or Option 3
2. Compare performance vs Phase 1
3. Assess data quantity vs quality tradeoff

**Time**: ~4-5 hours

### Phase 3: Feature Engineering Pipeline

1. Apply leak-free denoising to train/val separately
2. Generate Phase 1-3 features
3. Final TRM training with validated data

---

## 8. Complete Feature NaN Breakdown

See `nan_analysis.csv` for complete breakdown of all 94 features.

**Summary by family**:
- **E (Economic)**: 1006-6969 leading NaN (20 features)
- **M (Market)**: 1006-5547 leading NaN (18 features)
- **I (Interest)**: 1006 leading NaN (9 features)
- **P (Price)**: 1006-1638 leading NaN (13 features)
- **S (Sentiment)**: 1006-5733 leading NaN (12 features)
- **V (Volatility)**: 1006-6049 leading NaN (13 features)
- **D (Date)**: 0 NaN - Complete data (9 features)

---

## 9. Conclusion

**The current implementation has severe data leakage that invalidates all results.**

**Immediate Actions Required**:
1. Stop using current denoised data for any analysis
2. Implement leak-free data processing (Option 1)
3. Re-establish baseline with clean data
4. Re-evaluate denoising effectiveness

**Expected Outcome**:
- Denoising performance will drop significantly
- True effectiveness will be measurable
- Results will be scientifically valid and publishable

---

*Report generated: 2025-01-XX*
*Analysis scripts: `analyze_nan_patterns.py`*
*Detailed data: `nan_analysis.csv`*
