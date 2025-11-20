# Denoised Data Characteristics Analysis Report
Generated: 2025-11-13 13:31:57

---

## Executive Summary

### Signal-to-Noise Ratio
- **Average SNR Improvement**: 75992.72%
- ✅ **Significant improvement** in signal quality

### Feature Correlations
- Original: 0.1661
- Denoised: 0.1519
- Change: -0.0142

### Temporal Dependencies
- Original ACF (lag 1-10): 0.2633
- Denoised ACF (lag 1-10): 0.2485

### Predictability (Information Coefficient)
- **IC Improvement**: 16.39%
- ⚠️ **Marginal improvement** in predictive power

### Stationarity
- Original: 80.0% stationary features
- Denoised: 85.0% stationary features

---

## Detailed Results

See visualization files in `results/` directory:
- `snr_analysis.png`
- `correlation_analysis.png`
- `temporal_dependency_analysis.png`
- `predictability_analysis.png`
- `stationarity_analysis.png`

---

## Recommendations

### ⚠️ Denoising Shows Moderate Effect
- Marginal improvement in predictive power
- Consider ensemble approach (original + denoised)
- May need hyperparameter tuning for denoiser

