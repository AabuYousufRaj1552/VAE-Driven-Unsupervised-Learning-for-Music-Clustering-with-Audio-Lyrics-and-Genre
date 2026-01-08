"""
Visualization utilities for latent space and clustering results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_latent_space_2d(Z_2d, labels, title="Latent Space Visualization", 
                         output_path=None, figsize=(10, 8), cmap='tab10'):
    """Plot 2D latent space colored by cluster labels"""
    plt.figure(figsize=figsize)
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels, s=20, 
                         cmap=cmap, alpha=0.6, edgecolors='none')
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_training_curves(history_csv, output_path=None):
    """Plot VAE training loss curves"""
    df = pd.read_csv(history_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('VAE Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Component losses
    if 'train_recon' in df.columns and 'train_kl' in df.columns:
        axes[1].plot(df['epoch'], df['train_recon'], label='Reconstruction', linewidth=2)
        axes[1].plot(df['epoch'], df['train_kl'], label='KL Divergence', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss Component', fontsize=12)
        axes[1].set_title('Loss Components (Train)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_reconstruction_comparison(original, reconstructed, n_samples=5, 
                                   output_path=None, titles=None):
    """Plot original vs reconstructed spectrograms"""
    n_samples = min(n_samples, len(original))
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('VAE Reconstruction Quality', fontsize=16, fontweight='bold')
    
    for i in range(n_samples):
        # Original
        im1 = axes[i, 0].imshow(original[i], aspect='auto', origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'Original {i+1}' if titles is None else f'Original: {titles[i]}')
        axes[i, 0].set_ylabel('Frequency')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Reconstructed
        im2 = axes[i, 1].imshow(reconstructed[i], aspect='auto', origin='lower', cmap='viridis')
        axes[i, 1].set_title(f'Reconstructed {i+1}')
        plt.colorbar(im2, ax=axes[i, 1])
        
        if i == n_samples - 1:
            axes[i, 0].set_xlabel('Time')
            axes[i, 1].set_xlabel('Time')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_cluster_distribution_heatmap(labels, true_labels, label_names=None,
                                      output_path=None, figsize=(14, 10)):
    """Plot cluster vs true label distribution heatmap"""
    # Filter valid labels
    valid_mask = (labels != -1) & (true_labels != -1)
    labels_valid = labels[valid_mask]
    true_valid = true_labels[valid_mask]
    
    # Create contingency table
    contingency = pd.crosstab(labels_valid, true_valid)
    
    # Limit to top clusters and labels for readability
    top_clusters = contingency.sum(axis=1).nlargest(20).index
    top_labels = contingency.sum(axis=0).nlargest(30).index
    
    contingency_subset = contingency.loc[top_clusters, top_labels]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(contingency_subset.values, aspect='auto', cmap='YlOrRd')
    
    ax.set_xticks(np.arange(len(top_labels)))
    ax.set_yticks(np.arange(len(top_clusters)))
    
    if label_names:
        x_labels = [label_names.get(int(i), f"L{i}")[:15] for i in top_labels]
    else:
        x_labels = [f"Label {int(i)}" for i in top_labels]
    
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_yticklabels([f"C{int(c)}" for c in top_clusters], fontsize=8)
    
    ax.set_xlabel("True Label", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title("Cluster-Label Distribution Heatmap", fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Track Count")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_metrics_comparison(results_df, output_path=None, figsize=(14, 10)):
    """Plot comparison of clustering metrics across methods"""
    metrics_to_plot = []
    for m in ['silhouette', 'ari', 'nmi', 'purity', 'davies_bouldin']:
        if m in results_df.columns and not results_df[m].isna().all():
            metrics_to_plot.append(m)
    
    n_metrics = len(metrics_to_plot)
    if n_metrics == 0:
        print("No valid metrics to plot")
        return
    
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Clustering Metrics Comparison", fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        valid_data = results_df[~results_df[metric].isna()].copy()
        if len(valid_data) > 0:
            valid_data = valid_data.sort_values(by=metric, ascending=(metric == 'davies_bouldin'))
            
            bars = ax.barh(range(len(valid_data)), valid_data[metric], 
                          color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(valid_data)))
            ax.set_yticklabels(valid_data['method'], fontsize=8)
            ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Highlight best
            bars[0].set_color('darkgreen')
            bars[0].set_alpha(0.9)
    
    # Hide empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def save_results_summary(results_df, output_dir, prefix=""):
    """Save comprehensive results summary"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{prefix}clustering_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")
    
    # Save text summary
    txt_path = os.path.join(output_dir, f"{prefix}summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLUSTERING RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("BEST METHODS BY METRIC\n")
        f.write("=" * 80 + "\n\n")
        
        for metric in ['silhouette', 'ari', 'nmi', 'purity']:
            if metric in results_df.columns:
                valid = results_df[~results_df[metric].isna()]
                if len(valid) > 0:
                    ascending = (metric == 'davies_bouldin')
                    best = valid.sort_values(by=metric, ascending=ascending).iloc[0]
                    f.write(f"{metric.upper():20s}: {best['method']} ({best[metric]:.4f})\n")
    
    print(f"Saved summary: {txt_path}")
