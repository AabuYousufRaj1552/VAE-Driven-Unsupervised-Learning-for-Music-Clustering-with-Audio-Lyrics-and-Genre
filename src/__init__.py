"""
Package initialization for src module
"""

from .vae import (
    MLPVAE,
    ConvVAE,
    MultiModalVAE,
    SimpleAutoencoder,
    vae_loss,
    multimodal_vae_loss
)

from .dataset import (
    load_audio_2d,
    load_lyrics_embeddings,
    load_genre_labels,
    align_multimodal_data,
    NumpyDataset,
    AudioDataset,
    MultiModalDatasetLazy
)

from .clustering import (
    run_kmeans,
    run_agglomerative,
    run_dbscan_grid_search,
    run_all_clusterers,
    reduce_dimensions_pca,
    reduce_dimensions_2d
)

from .evaluation import (
    cluster_purity,
    compute_all_metrics,
    safe_silhouette,
    compare_methods,
    print_metrics_summary
)

__all__ = [
    # VAE models
    'MLPVAE',
    'ConvVAE',
    'MultiModalVAE',
    'SimpleAutoencoder',
    'vae_loss',
    'multimodal_vae_loss',
    # Dataset
    'load_audio_2d',
    'load_lyrics_embeddings',
    'load_genre_labels',
    'align_multimodal_data',
    'NumpyDataset',
    'AudioDataset',
    'MultiModalDatasetLazy',
    # Clustering
    'run_kmeans',
    'run_agglomerative',
    'run_dbscan_grid_search',
    'run_all_clusterers',
    'reduce_dimensions_pca',
    'reduce_dimensions_2d',
    # Evaluation
    'cluster_purity',
    'compute_all_metrics',
    'safe_silhouette',
    'compare_methods',
    'print_metrics_summary',
]

__version__ = '1.0.0'
