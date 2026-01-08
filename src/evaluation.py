"""
Evaluation metrics for clustering quality assessment
Includes: Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI, Purity
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def cluster_purity(y_true, y_pred):
    """
    Compute cluster purity
    
    Purity = (1/N) * sum_k max_j |cluster_k ∩ class_j|
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster labels
    
    Returns:
        Purity score (0 to 1, higher is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Filter out invalid labels
    mask = (y_true >= 0) & (y_pred >= 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    contingency = pd.crosstab(y_pred, y_true)
    return np.sum(np.amax(contingency.values, axis=1)) / np.sum(contingency.values)


def compute_all_metrics(X, labels, y_true=None):
    """
    Compute all clustering quality metrics
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        y_true: True labels (optional, for ARI/NMI/Purity)
    
    Returns:
        Dictionary with all metrics
    """
    valid = labels != -1
    metrics = {
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "ari": np.nan,
        "nmi": np.nan,
        "purity": np.nan,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "n_noise": int(np.sum(labels == -1))
    }
    
    # Metrics requiring valid clusters
    if valid.sum() > 5:
        unique_labels = set(labels[valid])
        if len(unique_labels) > 1:
            try:
                metrics["silhouette"] = float(silhouette_score(X[valid], labels[valid]))
            except:
                pass
            
            try:
                metrics["davies_bouldin"] = float(davies_bouldin_score(X[valid], labels[valid]))
            except:
                pass
            
            try:
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(X[valid], labels[valid]))
            except:
                pass
    
    # Metrics requiring true labels
    if y_true is not None:
        mask = (y_true != -1) & valid
        if mask.sum() > 5:
            try:
                metrics["ari"] = float(adjusted_rand_score(y_true[mask], labels[mask]))
            except:
                pass
            
            try:
                metrics["nmi"] = float(normalized_mutual_info_score(y_true[mask], labels[mask]))
            except:
                pass
            
            try:
                metrics["purity"] = float(cluster_purity(y_true, labels))
            except:
                pass
    
    return metrics


def safe_silhouette(X, labels):
    """Safely compute silhouette score with error handling"""
    labels = np.asarray(labels)
    valid = labels != -1
    if valid.sum() < 5:
        return np.nan
    if len(np.unique(labels[valid])) < 2:
        return np.nan
    try:
        return float(silhouette_score(X[valid], labels[valid]))
    except:
        return np.nan


def compare_methods(results_list):
    """
    Compare multiple clustering methods
    
    Args:
        results_list: List of (method_name, labels, metrics_dict)
    
    Returns:
        DataFrame with comparison
    """
    rows = []
    for method_name, labels, metrics in results_list:
        row = {"method": method_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by primary metric (silhouette by default)
    if "silhouette" in df.columns:
        df = df.sort_values(by="silhouette", ascending=False)
    
    return df


def print_metrics_summary(metrics_dict, method_name=""):
    """Pretty print metrics summary"""
    if method_name:
        print(f"\n{'='*60}")
        print(f"Metrics for: {method_name}")
        print(f"{'='*60}")
    
    metric_names = {
        "silhouette": "Silhouette Score",
        "davies_bouldin": "Davies-Bouldin Index",
        "calinski_harabasz": "Calinski-Harabasz Index",
        "ari": "Adjusted Rand Index",
        "nmi": "Normalized Mutual Info",
        "purity": "Cluster Purity",
        "n_clusters": "Number of Clusters",
        "n_noise": "Noise Points"
    }
    
    for key, value in metrics_dict.items():
        display_name = metric_names.get(key, key)
        if isinstance(value, float) and not np.isnan(value):
            print(f"  {display_name:25s}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {display_name:25s}: {value}")
        else:
            print(f"  {display_name:25s}: N/A")
