"""
Clustering utilities for VAE-based music clustering
Includes: KMeans, Agglomerative, DBSCAN with parameter tuning
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def run_kmeans(X, n_clusters=6, n_init=20, random_state=42):
    """Run K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def run_agglomerative(X, n_clusters=6, linkage='ward'):
    """Run Agglomerative Clustering"""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X)
    return labels, agg


def run_dbscan_grid_search(X, eps_values, min_samples_values, metric_fn=None):
    """
    Run DBSCAN with grid search over parameters
    
    Args:
        X: Feature matrix
        eps_values: List of eps values to try
        min_samples_values: List of min_samples values to try
        metric_fn: Function to compute quality metric (e.g., silhouette_score)
    
    Returns:
        best_labels, best_params, best_score
    """
    best_score = -1
    best_labels = None
    best_params = (None, None)
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)
            
            if metric_fn is not None:
                valid = labels != -1
                if valid.sum() > 5 and len(set(labels[valid])) > 1:
                    try:
                        score = metric_fn(X[valid], labels[valid])
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_params = (eps, min_samples)
                    except:
                        continue
    
    if best_labels is None:
        best_labels = np.full(len(X), -1)
        best_score = np.nan
    
    return best_labels, best_params, best_score


def run_all_clusterers(X, y_true=None, n_clusters=6, prefix="", 
                       kmeans_n_init=20, dbscan_eps_values=None, 
                       dbscan_min_samples_values=None, 
                       compute_metrics_fn=None, random_state=42):
    """
    Run all clustering algorithms and return results
    
    Args:
        X: Feature matrix
        y_true: True labels (optional, for ARI/NMI)
        n_clusters: Number of clusters for KMeans/Agglomerative
        prefix: Prefix for method names
        compute_metrics_fn: Function to compute all metrics
    
    Returns:
        List of (method_name, labels, metrics_dict)
    """
    if dbscan_eps_values is None:
        dbscan_eps_values = [0.5, 1.0, 2.0, 3.0]
    if dbscan_min_samples_values is None:
        dbscan_min_samples_values = [3, 5, 8]
    
    results = []
    
    # KMeans
    labels_km, _ = run_kmeans(X, n_clusters=n_clusters, n_init=kmeans_n_init, random_state=random_state)
    if compute_metrics_fn:
        metrics_km = compute_metrics_fn(X, labels_km, y_true)
        results.append((prefix + "KMeans", labels_km, metrics_km))
    else:
        results.append((prefix + "KMeans", labels_km, {}))
    
    # Agglomerative
    labels_ag, _ = run_agglomerative(X, n_clusters=n_clusters)
    if compute_metrics_fn:
        metrics_ag = compute_metrics_fn(X, labels_ag, y_true)
        results.append((prefix + "Agglomerative", labels_ag, metrics_ag))
    else:
        results.append((prefix + "Agglomerative", labels_ag, {}))
    
    # DBSCAN with grid search
    from sklearn.metrics import silhouette_score
    labels_db, db_params, db_score = run_dbscan_grid_search(
        X, dbscan_eps_values, dbscan_min_samples_values, 
        metric_fn=silhouette_score
    )
    
    if compute_metrics_fn:
        metrics_db = compute_metrics_fn(X, labels_db, y_true)
    else:
        metrics_db = {}
    
    method_name = prefix + f"DBSCAN(eps={db_params[0]},ms={db_params[1]})" if db_params[0] else prefix + "DBSCAN(failed)"
    results.append((method_name, labels_db, metrics_db))
    
    return results


def reduce_dimensions_pca(X, n_components=16, random_state=42):
    """Reduce dimensionality using PCA"""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def reduce_dimensions_2d(X, method="umap", random_state=42, n_neighbors=15, min_dist=0.1, perplexity=30):
    """
    Reduce to 2D for visualization using UMAP or t-SNE
    
    Args:
        X: Feature matrix
        method: "umap" or "tsne"
        random_state: Random seed
        n_neighbors, min_dist: UMAP parameters
        perplexity: t-SNE parameter
    
    Returns:
        X_2d: (N, 2) array
    """
    if method == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
        return reducer.fit_transform(X)
    else:
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca"
        )
        return tsne.fit_transform(X)
