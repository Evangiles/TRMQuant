"""
Feature Clustering Script

Analyzes train.csv features and creates statistical clusters
for group-specific multivariate denoising.

Usage:
    python train/cluster_features.py --k_min 3 --k_max 12 --visualize
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.clustering.feature_analyzer import FeatureAnalyzer
from models.clustering.cluster_manager import ClusterManager


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Extract feature column names from dataframe.

    Assumes target columns are: target, target_2d, target_5d
    All other numeric columns are features.

    EXCLUDES D* features (D1-D9): Binary/dummy data, no denoising needed.

    Args:
        df: Input dataframe

    Returns:
        List of feature column names (excluding D* features)
    """
    target_cols = ['target', 'target_2d', 'target_5d']

    # CRITICAL: Exclude forward-looking features to prevent data leakage
    leakage_cols = [
        'forward_returns',
        'risk_free_rate',
        'market_forward_excess_returns'
    ]

    # EXCLUDE D* features (binary/dummy data, keep original)
    d_features = [f'D{i}' for i in range(1, 10)]  # D1-D9

    exclude_cols = target_cols + ['date_id'] + leakage_cols + d_features
    all_cols = df.columns.tolist()

    # Remove target columns, date_id, leakage features, D* features, and non-numeric columns
    feature_cols = [
        col for col in all_cols
        if col not in exclude_cols and df[col].dtype in ['float64', 'int64']
    ]

    return feature_cols


def main():
    parser = argparse.ArgumentParser(
        description='Cluster features for group-specific denoising'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='TinyRecursiveModels/CSVs/train_only.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='FinancialDenoising/clustering_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--k_min',
        type=int,
        default=3,
        help='Minimum number of clusters'
    )
    parser.add_argument(
        '--k_max',
        type=int,
        default=12,
        help='Maximum number of clusters'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=60,
        help='Window size for rolling statistics'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='elbow',
        choices=['elbow', 'silhouette'],
        help='Method for optimal K selection'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("FEATURE CLUSTERING PIPELINE")
    print("="*80)
    print(f"\nData path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"K range: [{args.k_min}, {args.k_max}]")
    print(f"Method: {args.method}")
    print(f"Window size: {args.window_size}")
    print(f"Random state: {args.random_state}")

    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    print("\n" + "─"*80)
    print("STEP 1: Loading Data")
    print("─"*80)

    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} rows")

    feature_cols = get_feature_columns(df)
    print(f"✓ Found {len(feature_cols)} feature columns")
    print(f"  Features: {', '.join(feature_cols[:10])}")
    if len(feature_cols) > 10:
        print(f"           ... and {len(feature_cols)-10} more")

    # -------------------------------------------------------------------------
    # Step 2: Feature Analysis
    # -------------------------------------------------------------------------
    print("\n" + "─"*80)
    print("STEP 2: Analyzing Feature Characteristics")
    print("─"*80)

    analyzer = FeatureAnalyzer(window_size=args.window_size)
    analysis_df = analyzer.analyze_dataframe(df, feature_cols)

    print(f"✓ Analyzed {len(analysis_df)} features")

    # Save analysis results
    analysis_path = output_dir / 'feature_analysis.csv'
    analysis_df.to_csv(analysis_path, index=False)
    print(f"✓ Saved analysis to {analysis_path}")

    # Print summary
    analyzer.print_feature_summary(analysis_df)

    # -------------------------------------------------------------------------
    # Step 3: Feature Matrix Normalization
    # -------------------------------------------------------------------------
    print("\n" + "─"*80)
    print("STEP 3: Creating Normalized Feature Matrix")
    print("─"*80)

    feature_matrix, scaler = analyzer.get_feature_matrix(analysis_df)
    print(f"✓ Feature matrix shape: {feature_matrix.shape}")
    print(f"  (n_features={feature_matrix.shape[0]}, n_characteristics={feature_matrix.shape[1]})")

    # -------------------------------------------------------------------------
    # Step 4: Optimal K Selection
    # -------------------------------------------------------------------------
    print("\n" + "─"*80)
    print("STEP 4: Finding Optimal Number of Clusters")
    print("─"*80)

    manager = ClusterManager(
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state
    )

    viz_path = output_dir / 'k_selection.png' if args.visualize else None
    optimal_k = manager.find_optimal_k(
        feature_matrix,
        method=args.method,
        visualize=args.visualize,
        save_path=viz_path
    )

    # -------------------------------------------------------------------------
    # Step 5: Cluster Fitting
    # -------------------------------------------------------------------------
    print("\n" + "─"*80)
    print("STEP 5: Fitting Clusters")
    print("─"*80)

    cluster_labels = manager.fit_clusters(feature_matrix, k=optimal_k)

    # Print cluster summary
    manager.print_cluster_summary(feature_cols, analysis_df)

    # -------------------------------------------------------------------------
    # Step 6: Save Results
    # -------------------------------------------------------------------------
    print("\n" + "─"*80)
    print("STEP 6: Saving Results")
    print("─"*80)

    # Save cluster assignments
    cluster_json_path = output_dir / 'cluster_assignments.json'
    manager.save_cluster_assignments(
        feature_cols,
        cluster_json_path,
        analysis_df
    )

    # Save cluster labels as CSV for easy inspection
    cluster_df = pd.DataFrame({
        'feature_name': feature_cols,
        'cluster_id': cluster_labels
    })
    cluster_csv_path = output_dir / 'cluster_labels.csv'
    cluster_df.to_csv(cluster_csv_path, index=False)
    print(f"✓ Saved cluster labels to {cluster_csv_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("CLUSTERING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\n📊 Results:")
    print(f"  - Optimal K: {optimal_k}")
    print(f"  - Total features: {len(feature_cols)}")
    print(f"  - Features per cluster:")

    for cluster_id in range(optimal_k):
        n_features = (cluster_labels == cluster_id).sum()
        print(f"    Cluster {cluster_id}: {n_features} features ({100*n_features/len(feature_cols):.1f}%)")

    print(f"\n📁 Output files:")
    print(f"  - Feature analysis: {analysis_path}")
    print(f"  - Cluster assignments: {cluster_json_path}")
    print(f"  - Cluster labels: {cluster_csv_path}")
    if args.visualize:
        print(f"  - K-selection plot: {viz_path}")

    print("\n✅ Pipeline completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
