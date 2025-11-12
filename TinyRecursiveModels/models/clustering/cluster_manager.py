"""
Cluster Manager for Feature Grouping

Performs K-means clustering on feature characteristics
and manages cluster assignments for group-specific denoising.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class ClusterManager:
    """
    Manages feature clustering using K-means algorithm.

    Features:
    - Elbow method for optimal K selection
    - Silhouette score for cluster quality assessment
    - Cluster assignment saving/loading
    - Cluster characteristics analysis
    """

    def __init__(
        self,
        k_min: int = 3,
        k_max: int = 12,
        random_state: int = 42
    ):
        """
        Args:
            k_min: Minimum number of clusters to consider
            k_max: Maximum number of clusters to consider
            random_state: Random seed for reproducibility
        """
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

        self.optimal_k: Optional[int] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.inertias: Dict[int, float] = {}
        self.silhouette_scores: Dict[int, float] = {}

    def find_optimal_k(
        self,
        feature_matrix: np.ndarray,
        method: str = 'elbow',
        visualize: bool = True,
        save_path: Optional[Path] = None
    ) -> int:
        """
        Find optimal number of clusters using elbow method or silhouette score.

        Args:
            feature_matrix: Normalized feature matrix (n_features, n_characteristics)
            method: 'elbow' or 'silhouette'
            visualize: Whether to plot results
            save_path: Path to save visualization

        Returns:
            Optimal K value
        """
        print(f"\nFinding optimal K using {method} method...")
        print(f"Testing K from {self.k_min} to {self.k_max}")

        k_range = range(self.k_min, self.k_max + 1)

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(feature_matrix)

            self.inertias[k] = kmeans.inertia_
            self.silhouette_scores[k] = silhouette_score(feature_matrix, labels)

            print(f"  K={k}: Inertia={kmeans.inertia_:.4f}, Silhouette={self.silhouette_scores[k]:.4f}")

        if method == 'elbow':
            self.optimal_k = self._find_elbow_point()
        else:  # silhouette
            self.optimal_k = max(self.silhouette_scores, key=self.silhouette_scores.get)

        print(f"\n✓ Optimal K selected: {self.optimal_k}")

        if visualize:
            self._visualize_k_selection(save_path)

        return self.optimal_k

    def _find_elbow_point(self) -> int:
        """
        Find elbow point in inertia curve using second derivative.

        Returns:
            K value at elbow point
        """
        k_values = sorted(self.inertias.keys())
        inertias = [self.inertias[k] for k in k_values]

        # Compute second derivative
        second_derivative = np.diff(inertias, 2)

        # Find maximum curvature (elbow point)
        # Add 2 because we lost 2 points in double differentiation
        elbow_idx = np.argmax(second_derivative) + 2

        return k_values[elbow_idx]

    def _visualize_k_selection(self, save_path: Optional[Path] = None):
        """
        Visualize elbow curve and silhouette scores.

        Args:
            save_path: Path to save figure
        """
        k_values = sorted(self.inertias.keys())
        inertias = [self.inertias[k] for k in k_values]
        silhouettes = [self.silhouette_scores[k] for k in k_values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow curve
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(self.optimal_k, color='r', linestyle='--', label=f'Optimal K={self.optimal_k}')
        ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Silhouette scores
        ax2.plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
        ax2.axvline(self.optimal_k, color='r', linestyle='--', label=f'Optimal K={self.optimal_k}')
        ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ K-selection plot saved to {save_path}")

        plt.close()

    def fit_clusters(
        self,
        feature_matrix: np.ndarray,
        k: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit K-means clustering with specified or optimal K.

        Args:
            feature_matrix: Normalized feature matrix
            k: Number of clusters (uses optimal_k if None)

        Returns:
            Cluster labels for each feature
        """
        if k is None:
            if self.optimal_k is None:
                raise ValueError("Must call find_optimal_k() first or provide k")
            k = self.optimal_k

        print(f"\nFitting K-means with K={k}...")

        self.kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=20,
            max_iter=500
        )

        self.cluster_labels = self.kmeans.fit_predict(feature_matrix)

        # Quality metrics
        silhouette = silhouette_score(feature_matrix, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(feature_matrix, self.cluster_labels)

        print(f"✓ Clustering completed")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")

        return self.cluster_labels

    def save_cluster_assignments(
        self,
        feature_names: List[str],
        output_path: Path,
        analysis_df: Optional[pd.DataFrame] = None
    ):
        """
        Save cluster assignments to JSON file.

        Args:
            feature_names: List of feature names
            output_path: Path to save JSON file
            analysis_df: Optional DataFrame with feature characteristics
        """
        if self.cluster_labels is None:
            raise ValueError("Must call fit_clusters() first")

        # Create cluster assignment dictionary
        cluster_dict = {
            'metadata': {
                'n_clusters': int(self.optimal_k or len(np.unique(self.cluster_labels))),
                'n_features': len(feature_names),
                'random_state': self.random_state
            },
            'clusters': {}
        }

        # Group features by cluster
        for cluster_id in range(cluster_dict['metadata']['n_clusters']):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_features = [
                feature_names[i] for i in range(len(feature_names))
                if cluster_mask[i]
            ]

            cluster_info = {
                'feature_count': len(cluster_features),
                'features': cluster_features
            }

            # Add cluster characteristics if analysis_df provided
            if analysis_df is not None:
                cluster_features_df = analysis_df[
                    analysis_df['feature_name'].isin(cluster_features)
                ]

                cluster_info['characteristics'] = {
                    'mean_hurst': float(cluster_features_df['hurst_exponent'].mean()),
                    'mean_adf_pvalue': float(cluster_features_df['adf_pvalue'].mean()),
                    'mean_volatility': float(cluster_features_df['volatility_mean'].mean()),
                    'mean_autocorr': float(cluster_features_df['autocorr_lag1'].mean()),
                    'mean_skewness': float(cluster_features_df['skewness'].mean()),
                    'mean_kurtosis': float(cluster_features_df['kurtosis'].mean())
                }

                # Determine cluster type
                hurst = cluster_info['characteristics']['mean_hurst']
                if hurst < 0.45:
                    cluster_type = 'mean_reverting'
                elif hurst > 0.55:
                    cluster_type = 'trending'
                else:
                    cluster_type = 'random_walk'

                cluster_info['type'] = cluster_type

            cluster_dict['clusters'][f'cluster_{cluster_id}'] = cluster_info

        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(cluster_dict, f, indent=2)

        print(f"\n✓ Cluster assignments saved to {output_path}")

    def load_cluster_assignments(self, input_path: Path) -> Dict:
        """
        Load cluster assignments from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Cluster assignment dictionary
        """
        with open(input_path, 'r') as f:
            cluster_dict = json.load(f)

        print(f"\n✓ Loaded cluster assignments from {input_path}")
        print(f"  Number of clusters: {cluster_dict['metadata']['n_clusters']}")
        print(f"  Total features: {cluster_dict['metadata']['n_features']}")

        return cluster_dict

    def print_cluster_summary(
        self,
        feature_names: List[str],
        analysis_df: Optional[pd.DataFrame] = None
    ):
        """
        Print detailed cluster summary.

        Args:
            feature_names: List of feature names
            analysis_df: Optional DataFrame with feature characteristics
        """
        if self.cluster_labels is None:
            raise ValueError("Must call fit_clusters() first")

        n_clusters = len(np.unique(self.cluster_labels))

        print("\n" + "="*80)
        print("CLUSTERING SUMMARY")
        print("="*80)

        for cluster_id in range(n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_features = [
                feature_names[i] for i in range(len(feature_names))
                if cluster_mask[i]
            ]

            print(f"\n{'─'*80}")
            print(f"CLUSTER {cluster_id}")
            print(f"{'─'*80}")
            print(f"Number of features: {len(cluster_features)}")

            if analysis_df is not None:
                cluster_df = analysis_df[analysis_df['feature_name'].isin(cluster_features)]

                print(f"\nCharacteristics:")
                print(f"  Hurst Exponent:  {cluster_df['hurst_exponent'].mean():.4f} ± {cluster_df['hurst_exponent'].std():.4f}")
                print(f"  ADF p-value:     {cluster_df['adf_pvalue'].mean():.4f} ± {cluster_df['adf_pvalue'].std():.4f}")
                print(f"  Volatility:      {cluster_df['volatility_mean'].mean():.4f} ± {cluster_df['volatility_mean'].std():.4f}")
                print(f"  Autocorr (lag1): {cluster_df['autocorr_lag1'].mean():.4f} ± {cluster_df['autocorr_lag1'].std():.4f}")
                print(f"  Skewness:        {cluster_df['skewness'].mean():.4f} ± {cluster_df['skewness'].std():.4f}")
                print(f"  Kurtosis:        {cluster_df['kurtosis'].mean():.4f} ± {cluster_df['kurtosis'].std():.4f}")

                # Determine cluster type
                hurst = cluster_df['hurst_exponent'].mean()
                if hurst < 0.45:
                    cluster_type = "Mean-Reverting (H<0.45)"
                elif hurst > 0.55:
                    cluster_type = "Trending (H>0.55)"
                else:
                    cluster_type = "Random Walk (0.45≤H≤0.55)"

                print(f"\n  Type: {cluster_type}")

            print(f"\nFeatures: {', '.join(cluster_features[:10])}")
            if len(cluster_features) > 10:
                print(f"          ... and {len(cluster_features)-10} more")

        print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage
    print("ClusterManager module loaded successfully")
    print("\nExample usage:")
    print("  manager = ClusterManager(k_min=3, k_max=12)")
    print("  optimal_k = manager.find_optimal_k(feature_matrix, visualize=True)")
    print("  labels = manager.fit_clusters(feature_matrix)")
    print("  manager.save_cluster_assignments(feature_names, Path('clusters.json'), analysis_df)")
    print("  manager.print_cluster_summary(feature_names, analysis_df)")
