"""
Dataset utilities for loading and preprocessing audio, lyrics, and genre data
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def load_audio_2d(npz_path, key="mfcc", max_time=500):
    """Load and optionally truncate/pad audio features"""
    d = np.load(npz_path)
    if key not in d:
        raise KeyError(f"Key '{key}' not in {npz_path}. Available: {list(d.keys())}")
    feat = d[key].astype(np.float32)
    F, T = feat.shape
    if T > max_time:
        feat = feat[:, :max_time]
    elif T < max_time:
        pad = np.zeros((F, max_time - T), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=1)
    return feat


def load_lyrics_embeddings(lyrics_csv, sbert_npy=None, tfidf_npz=None):
    """Load lyrics embeddings (SBERT or TF-IDF)"""
    lyrics_df = pd.read_csv(lyrics_csv)
    lyrics_df["track_id"] = lyrics_df["track_id"].astype(str)
    lyrics_ids = lyrics_df["track_id"].tolist()
    
    if sbert_npy and os.path.exists(sbert_npy):
        lyrics_emb = np.load(sbert_npy).astype(np.float32)
        lyrics_repr = "sbert"
        lyrics_lookup = {tid: lyrics_emb[i] for i, tid in enumerate(lyrics_ids)}
    elif tfidf_npz and os.path.exists(tfidf_npz):
        from scipy.sparse import load_npz
        lyrics_emb = load_npz(tfidf_npz)
        lyrics_repr = "tfidf"
        lyrics_lookup = {tid: i for i, tid in enumerate(lyrics_ids)}
    else:
        raise RuntimeError("No lyrics embeddings found")
    
    return lyrics_lookup, lyrics_emb, lyrics_repr, lyrics_df


def load_genre_labels(genre_csv):
    """Load genre multi-hot and single-label encodings"""
    genre_df = pd.read_csv(genre_csv)
    genre_id_col = None
    for c in genre_df.columns:
        if c.lower() in ["track_id", "song_id", "music_id", "id"]:
            genre_id_col = c
            break
    
    genre_cols = [c for c in genre_df.columns if c.startswith("genre__")]
    if not genre_id_col or not genre_cols:
        raise RuntimeError("Genre file must have ID column and genre__* columns")
    
    genre_df[genre_id_col] = genre_df[genre_id_col].astype(str)
    
    genre_multi_lookup = {}
    genre_label_lookup = {}
    for _, r in genre_df[[genre_id_col] + genre_cols].iterrows():
        tid = str(r[genre_id_col])
        v = r[genre_cols].values.astype(np.float32)
        genre_multi_lookup[tid] = v
        genre_label_lookup[tid] = int(np.argmax(v)) if v.sum() > 0 else -1
    
    return genre_multi_lookup, genre_label_lookup, len(genre_cols)


class NumpyDataset(Dataset):
    """Simple dataset for numpy arrays"""
    
    def __init__(self, X_np):
        self.X = torch.from_numpy(X_np.astype(np.float32))
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx]


class AudioDataset(Dataset):
    """Dataset for audio features"""
    
    def __init__(self, X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


class MultiModalDatasetLazy(Dataset):
    """Memory-efficient dataset with on-the-fly audio loading"""
    
    def __init__(self, indices, kept_ids, audio_map, X_lyrics, X_genre, audio_key, F, T):
        self.indices = indices
        self.kept_ids = kept_ids
        self.audio_map = audio_map
        self.X_lyrics = X_lyrics
        self.X_genre = X_genre
        self.audio_key = audio_key
        self.F = F
        self.T = T
        self.zero_audio = np.zeros((1, F, T), dtype=np.float32)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        tid = self.kept_ids[real_idx]
        
        if tid in self.audio_map:
            audio = load_audio_2d(self.audio_map[tid], self.audio_key, max_time=self.T)[np.newaxis, :]
        else:
            audio = self.zero_audio.copy()
        
        lyrics = self.X_lyrics[real_idx]
        genre = self.X_genre[real_idx]
        
        return audio, lyrics, genre


def align_multimodal_data(audio_dir, lyrics_csv, genre_csv, 
                          sbert_npy=None, tfidf_npz=None, 
                          audio_key="mfcc", keep_missing=True):
    """
    Align audio, lyrics, and genre data across track IDs
    
    Returns:
        kept_ids, X_lyrics, X_genre, y_genre, audio_map, F, T
    """
    # Load audio files
    audio_npz_files = sorted(glob.glob(os.path.join(audio_dir, "*.npz")))
    audio_ids = [os.path.splitext(os.path.basename(f))[0] for f in audio_npz_files]
    audio_map = dict(zip(audio_ids, audio_npz_files))
    
    # Get audio shape
    sample_audio = load_audio_2d(audio_map[audio_ids[0]], audio_key)
    F, T = sample_audio.shape
    
    # Load lyrics
    lyrics_lookup, lyrics_emb, lyrics_repr, lyrics_df = load_lyrics_embeddings(
        lyrics_csv, sbert_npy, tfidf_npz
    )
    lyrics_dim = int(lyrics_emb.shape[1])
    
    # Load genre
    genre_multi_lookup, genre_label_lookup, genre_dim = load_genre_labels(genre_csv)
    
    # Align
    all_ids = sorted(list(set(audio_ids) | set(lyrics_lookup.keys()) | set(genre_multi_lookup.keys())))
    kept_ids = []
    
    for tid in all_ids:
        ha = tid in audio_map
        hl = tid in lyrics_lookup
        hg = tid in genre_multi_lookup
        if keep_missing or (ha and hl and hg):
            kept_ids.append(tid)
    
    # Build arrays
    X_lyrics = np.zeros((len(kept_ids), lyrics_dim), dtype=np.float32)
    X_genre = np.zeros((len(kept_ids), genre_dim), dtype=np.float32)
    y_genre = np.full((len(kept_ids),), -1, dtype=np.int32)
    
    for i, tid in enumerate(kept_ids):
        if lyrics_repr == "sbert":
            lv = lyrics_lookup.get(tid, None)
        else:
            idx = lyrics_lookup.get(tid, None)
            if idx is not None:
                lv = lyrics_emb.getrow(idx).toarray().reshape(-1).astype(np.float32)
            else:
                lv = None
        
        if lv is not None:
            X_lyrics[i] = lv
        
        gv = genre_multi_lookup.get(tid, None)
        if gv is not None:
            X_genre[i] = gv
            y_genre[i] = genre_label_lookup.get(tid, -1)
    
    # Normalize lyrics
    scaler = StandardScaler()
    X_lyrics = scaler.fit_transform(X_lyrics).astype(np.float32)
    
    return kept_ids, X_lyrics, X_genre, y_genre, audio_map, F, T, lyrics_dim, genre_dim
