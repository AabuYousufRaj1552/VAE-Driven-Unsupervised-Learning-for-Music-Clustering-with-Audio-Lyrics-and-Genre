# Hybrid Music Clustering using Variational Autoencoders

Unsupervised learning pipeline for clustering hybrid language (English/Bangla) music tracks using Variational Autoencoders (VAE) with multi-modal features (audio + lyrics + genre).

## 📋 Project Overview

This project implements VAE-based clustering for music analysis with:
- **Multi-modal feature extraction**: Audio (MFCC/Mel-spectrogram), Lyrics (SBERT embeddings), Genre labels
- **Multiple VAE architectures**: MLP-VAE, ConvVAE, Conditional VAE (CVAE), Beta-VAE
- **Clustering algorithms**: K-Means, Agglomerative Clustering, DBSCAN
- **Comprehensive evaluation**: Silhouette Score, ARI, NMI, Cluster Purity, Davies-Bouldin Index, Calinski-Harabasz Index

## 🏗️ Repository Structure

```
project/
│
├── data/ 
|   |── dataset_link
│       ├── audio/               ← Raw audio files
│       ├── lyrics/              ← Raw lyrics data
│       ├── Audio_Features/      ← Extracted features from audio (e.g., MFCC, spectrograms)
│       ├── Lyrics_Preprocessed/ ← Preprocessed lyrics (e.g., tokenized)
│       ├── MultiModal/          ← Multimodal data (audio + lyrics)
│       ├── audio_metadata/      ← Metadata for audio files
│       ├── lyrics_metadata/     ← Metadata for lyrics
│       ├── genre_label_classes/ ← Genre label classes
│       ├── genre_processed/     ← Processed genre data (e.g., encoded)
│       └── genre/               ← Raw genre data
│
├── notebooks/
│   ├── all_in_one_scattered     ← Integrated notebook for various tasks
│   ├── Create_Genre            ← Genre creation and modeling notebook
│   ├── exploratory             ← Exploratory data analysis (EDA) notebook
│   └── generate_eda_visualizations ← Visualizations for EDA
│
├── results/
│   ├── eda/                    ← Results from exploratory data analysis
│   ├── EasyTask/               ← Results for easy tasks
│   ├── MediumTask/             ← Results for medium tasks
│   ├── MediumTask_WithARI/     ← Results for medium tasks with ARI 
│   ├── HardTask/               ← Results for hard tasks
│   └── HardTask_CVAE/          ← Results for hard tasks using CVAE 
│
└── src/                         ← Source code and project files
    ├── __init__.py             ← Marks this folder as a Python package
    ├── vae.py                  ← Variational Autoencoder model code
    ├── dataset.py              ← Dataset handling code
    ├── clustering.py           ← Clustering code (e.g., for genre or features)
    ├── evaluation.py           ← Model evaluation code
    ├── visualization.py        ← Visualization code
├── README.md               ← Project overview 
└── requirements.txt        ← List of project dependencies

```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster training)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Place your audio files in `Audio/` directory
2. Place lyrics files in `Lyrics/` directory
3. Place genre metadata in `genre.csv`

## 📊 Usage

### Preprocessing

Run the preprocessing cells in the notebook to:
1. Process genre labels and create multi-hot encodings
2. Extract audio features (MFCC, Mel-spectrograms)
3. Process and embed lyrics using SBERT
4. Align multi-modal data across track IDs

### Training

The project includes three difficulty levels:

#### Easy Task
- Basic MLP-VAE with mean+std audio features
- K-Means clustering
- Baseline: PCA + K-Means
- Metrics: Silhouette Score, Calinski-Harabasz Index

#### Medium Task
- ConvVAE for 2D audio features
- Hybrid features: Audio latent + Lyrics embeddings
- Multiple clustering algorithms (KMeans, Agglomerative, DBSCAN)
- Systematic hyperparameter tuning
- Metrics: Silhouette, Davies-Bouldin, ARI

#### Hard Task
- Conditional VAE (CVAE) with genre conditioning
- Beta-VAE for disentangled representations
- Multi-modal clustering (Audio + Lyrics + Genre)
- Complete metrics suite: Silhouette, ARI, NMI, Purity
- Enhanced visualizations

### Running Experiments

```python
# Import modules
from src.vae import ConvVAE, MultiModalVAE
from src.dataset import align_multimodal_data
from src.clustering import run_all_clusterers
from src.evaluation import compute_all_metrics

# See notebooks/exploratory.ipynb for complete examples
```

## 📈 Results

Results are saved in the `Results/` directory:
- **EasyTask/**: Basic VAE results, latent features, cluster assignments
- **MediumTask_WithARI/**: ConvVAE results with ARI computation
- **HardTask_CVAE/**: CVAE/Beta-VAE results with complete metrics

Key outputs:
- `clustering_results.csv`: Comparison of all methods
- `cluster_assignments.csv`: Track-level cluster labels
- `vae_training_history.csv`: Training loss curves
- Visualization plots in respective directories

## 📊 Evaluation Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|---------|
| Silhouette Score | Cluster cohesion vs separation | [-1, 1] | Higher |
| Davies-Bouldin | Average cluster similarity | [0, ∞) | Lower |
| Calinski-Harabasz | Between/within variance ratio | [0, ∞) | Higher |
| ARI | Agreement with true labels | [-1, 1] | Higher |
| NMI | Mutual information with labels | [0, 1] | Higher |
| Purity | Dominant class in cluster | [0, 1] | Higher |

## 🔧 Configuration

Key hyperparameters can be adjusted:
- `LATENT_DIM`: VAE latent dimension (16-128)
- `BETA`: KL divergence weight for Beta-VAE (0.1-8.0)
- `N_CLUSTERS`: Number of clusters (6-20)
- `EPOCHS`: Training epochs (20-60)
- `BATCH_SIZE`: Mini-batch size (16-64)

## 📝 Citation

If you use this code, please cite:
```
[Your Project Title]
[Your Name/Team]
[Year]
```

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email].

## 🙏 Acknowledgments

- SBERT for multilingual text embeddings
- PyTorch for deep learning framework
- Scikit-learn for clustering and evaluation
