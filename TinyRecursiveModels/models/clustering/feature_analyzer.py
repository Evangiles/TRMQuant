"""
Feature Analyzer for Time Series Clustering

Extracts statistical and time series characteristics from features
to enable clustering of heterogeneous financial features.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Extracts comprehensive statistical features for time series clustering.

    Features extracted:
    - Basic statistics: mean, std, min, max
    - Distribution shape: skewness, kurtosis
    - Stationarity: ADF test p-value
    - Long-term memory: Hurst exponent
    - Autocorrelation: lag-1 autocorrelation
    - Volatility: rolling std statistics
    """

    def __init__(self, window_size: int = 60):
        """
        Args:
            window_size: Rolling window size for volatility calculations
        """
        self.window_size = window_size

    def compute_hurst_exponent(self, series: np.ndarray) -> float:
        """
        Compute Hurst exponent using R/S analysis.

        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending

        Args:
            series: Time series data

        Returns:
            Hurst exponent value
        """
        if len(series) < 100:
            return 0.5  # Default for short series

        # Remove NaN values
        series = series[~np.isnan(series)]
        if len(series) < 100:
            return 0.5

        lags = range(2, min(100, len(series) // 2))
        tau = []

        for lag in lags:
            # Calculate standard deviation of differenced series
            std = np.std(np.subtract(series[lag:], series[:-lag]))
            tau.append(std)

        # Linear regression on log-log plot
        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5

    def compute_adf_pvalue(self, series: np.ndarray) -> float:
        """
        Compute Augmented Dickey-Fuller test p-value for stationarity.

        p < 0.05: Stationary
        p >= 0.05: Non-stationary

        Args:
            series: Time series data

        Returns:
            ADF test p-value
        """
        try:
            # Remove NaN values
            series_clean = series[~np.isnan(series)]
            if len(series_clean) < 20:
                return 1.0  # Cannot test, assume non-stationary

            result = adfuller(series_clean, maxlag=10, regression='c')
            return result[1]  # p-value
        except:
            return 1.0

    def analyze_feature(self, series: pd.Series) -> Dict[str, float]:
        """
        Extract all statistical features from a single time series.

        Args:
            series: Time series feature column

        Returns:
            Dictionary of extracted features
        """
        values = series.values

        # Remove NaN for calculations
        valid_values = values[~np.isnan(values)]

        if len(valid_values) < 10:
            # Return default values for insufficient data
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'adf_pvalue': 1.0,
                'hurst_exponent': 0.5,
                'autocorr_lag1': 0.0,
                'volatility_mean': 0.0,
                'volatility_std': 0.0
            }

        # Basic statistics
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)

        # Distribution shape
        skewness = stats.skew(valid_values)
        kurtosis_val = stats.kurtosis(valid_values)

        # Stationarity test
        adf_pval = self.compute_adf_pvalue(valid_values)

        # Long-term memory
        hurst = self.compute_hurst_exponent(valid_values)

        # Autocorrelation
        try:
            autocorr = pd.Series(valid_values).autocorr(lag=1)
            if np.isnan(autocorr):
                autocorr = 0.0
        except:
            autocorr = 0.0

        # Rolling volatility statistics
        try:
            rolling_std = pd.Series(valid_values).rolling(
                window=min(self.window_size, len(valid_values) // 2)
            ).std()
            vol_mean = np.nanmean(rolling_std)
            vol_std = np.nanstd(rolling_std)
        except:
            vol_mean = std_val
            vol_std = 0.0

        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(min_val),
            'max': float(max_val),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis_val),
            'adf_pvalue': float(adf_pval),
            'hurst_exponent': float(hurst),
            'autocorr_lag1': float(autocorr),
            'volatility_mean': float(vol_mean),
            'volatility_std': float(vol_std)
        }

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Analyze all features in a dataframe.

        Args:
            df: Input dataframe with time series features
            feature_cols: List of feature column names to analyze

        Returns:
            DataFrame where each row represents a feature's characteristics
            Columns: feature_name + all statistical features
        """
        results = []

        for col in feature_cols:
            if col not in df.columns:
                print(f"Warning: Column {col} not found in dataframe")
                continue

            features = self.analyze_feature(df[col])
            features['feature_name'] = col
            results.append(features)

        # Create DataFrame with feature_name as first column
        result_df = pd.DataFrame(results)
        cols = ['feature_name'] + [c for c in result_df.columns if c != 'feature_name']
        result_df = result_df[cols]

        return result_df

    def get_feature_matrix(self, analysis_df: pd.DataFrame) -> np.ndarray:
        """
        Convert analysis DataFrame to feature matrix for clustering.

        Args:
            analysis_df: Output from analyze_dataframe()

        Returns:
            Normalized feature matrix (n_features, n_characteristics)
        """
        # Exclude feature_name column
        feature_cols = [c for c in analysis_df.columns if c != 'feature_name']
        feature_df = analysis_df[feature_cols].copy()

        # Handle NaN and inf values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(feature_df.median())

        # If still NaN (all values were NaN), fill with 0
        feature_df = feature_df.fillna(0)

        feature_matrix = feature_df.values

        # Normalize to [0, 1] range for each characteristic
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized = scaler.fit_transform(feature_matrix)

        return normalized, scaler

    def print_feature_summary(self, analysis_df: pd.DataFrame):
        """
        Print summary statistics of analyzed features.

        Args:
            analysis_df: Output from analyze_dataframe()
        """
        print("\n" + "="*80)
        print("FEATURE ANALYSIS SUMMARY")
        print("="*80)

        print(f"\nTotal features analyzed: {len(analysis_df)}")

        # Stationarity distribution
        stationary = (analysis_df['adf_pvalue'] < 0.05).sum()
        print(f"\nStationarity (ADF test p<0.05):")
        print(f"  Stationary: {stationary} ({100*stationary/len(analysis_df):.1f}%)")
        print(f"  Non-stationary: {len(analysis_df)-stationary} ({100*(len(analysis_df)-stationary)/len(analysis_df):.1f}%)")

        # Hurst exponent distribution
        mean_reverting = (analysis_df['hurst_exponent'] < 0.5).sum()
        trending = (analysis_df['hurst_exponent'] > 0.5).sum()
        random = len(analysis_df) - mean_reverting - trending

        print(f"\nHurst Exponent Distribution:")
        print(f"  Mean-reverting (H<0.5): {mean_reverting} ({100*mean_reverting/len(analysis_df):.1f}%)")
        print(f"  Random walk (H≈0.5): {random} ({100*random/len(analysis_df):.1f}%)")
        print(f"  Trending (H>0.5): {trending} ({100*trending/len(analysis_df):.1f}%)")

        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Mean range: [{analysis_df['mean'].min():.4f}, {analysis_df['mean'].max():.4f}]")
        print(f"  Std range: [{analysis_df['std'].min():.4f}, {analysis_df['std'].max():.4f}]")
        print(f"  Skewness range: [{analysis_df['skewness'].min():.4f}, {analysis_df['skewness'].max():.4f}]")
        print(f"  Kurtosis range: [{analysis_df['kurtosis'].min():.4f}, {analysis_df['kurtosis'].max():.4f}]")

        print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage
    print("FeatureAnalyzer module loaded successfully")
    print("\nExample usage:")
    print("  analyzer = FeatureAnalyzer(window_size=60)")
    print("  analysis_df = analyzer.analyze_dataframe(df, feature_cols)")
    print("  feature_matrix, scaler = analyzer.get_feature_matrix(analysis_df)")
    print("  analyzer.print_feature_summary(analysis_df)")
