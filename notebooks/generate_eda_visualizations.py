"""
Generate EDA Visualizations for Research Report
Creates comprehensive exploratory data analysis plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150

# Create output directory
output_dir = Path("Results/EDA")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading data...")

# Load genre data
try:
    df_genre = pd.read_csv("genre_processed.csv")
    print(f"Loaded genre data: {len(df_genre)} tracks")
except:
    print("Warning: genre_processed.csv not found")
    df_genre = None

# Load audio metadata
try:
    df_audio = pd.read_csv("audio_metadata.csv")
    print(f"Loaded audio metadata: {len(df_audio)} tracks")
except:
    print("Warning: audio_metadata.csv not found")
    df_audio = None

# Load lyrics metadata
try:
    df_lyrics = pd.read_csv("lyrics_metadata.csv")
    print(f"Loaded lyrics metadata: {len(df_lyrics)} tracks")
except:
    print("Warning: lyrics_metadata.csv not found")
    df_lyrics = None

# Load genre label classes
try:
    with open("genre_label_classes.json", "r") as f:
        genre_classes = json.load(f)
    print(f"Loaded {len(genre_classes)} genre classes")
except:
    print("Warning: genre_label_classes.json not found")
    genre_classes = []

print("\n" + "="*60)
print("GENERATING EDA VISUALIZATIONS")
print("="*60 + "\n")

# ==========================================
# 1. GENRE DISTRIBUTION
# ==========================================
if df_genre is not None and genre_classes:
    print("1. Creating genre distribution plots...")
    
    # Count genres (multi-hot encoded)
    # Identify genre columns (numeric columns excluding Music_Id)
    genre_cols = [col for col in df_genre.columns 
                  if col not in ['Music_Id', 'Genres'] and df_genre[col].dtype in [np.int64, np.float64, 'int64', 'float64']]
    
    if genre_cols and len(genre_cols) > 0:
        genre_counts = df_genre[genre_cols].sum().astype(float).sort_values(ascending=False)
        
        # Top 20 genres
        fig, ax = plt.subplots(figsize=(14, 8))
        top_20 = genre_counts.head(20)
        bars = ax.barh(range(len(top_20)), top_20.values, color='steelblue', edgecolor='navy', alpha=0.7)
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels(top_20.index, fontsize=11)
        ax.set_xlabel('Number of Tracks', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Most Common Genres in Dataset', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_20.values)):
            ax.text(val + 1, i, f'{int(val)}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "genre_distribution_top20.png", bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: genre_distribution_top20.png")
        
        # Genre distribution histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(genre_counts.values, bins=30, color='coral', edgecolor='darkred', alpha=0.7)
        ax.set_xlabel('Number of Tracks', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Genres', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Genre Frequencies', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        ax.axvline(genre_counts.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {genre_counts.median():.0f}')
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / "genre_frequency_histogram.png", bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: genre_frequency_histogram.png")
        
        # Multi-genre statistics
        genres_per_track = df_genre[genre_cols].sum(axis=1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(genres_per_track, bins=range(1, int(genres_per_track.max())+2), 
                color='mediumseagreen', edgecolor='darkgreen', alpha=0.7)
        ax1.set_xlabel('Number of Genres per Track', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Tracks', fontsize=12, fontweight='bold')
        ax1.set_title('Multi-Genre Distribution', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        genre_counts_pie = genres_per_track.value_counts().sort_index()
        colors = plt.cm.Set3(range(len(genre_counts_pie)))
        ax2.pie(genre_counts_pie.values, labels=[f'{int(x)} genres' for x in genre_counts_pie.index],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Track Distribution by Genre Count', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "multi_genre_statistics.png", bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: multi_genre_statistics.png")

# ==========================================
# 2. LANGUAGE DISTRIBUTION
# ==========================================
if df_lyrics is not None and 'Language' in df_lyrics.columns:
    print("\n2. Creating language distribution plots...")
    
    lang_counts = df_lyrics['Language'].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    colors_lang = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(lang_counts)]
    bars = ax1.bar(range(len(lang_counts)), lang_counts.values, color=colors_lang, 
                   edgecolor='black', alpha=0.8)
    ax1.set_xticks(range(len(lang_counts)))
    ax1.set_xticklabels(lang_counts.index, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Tracks', fontsize=12, fontweight='bold')
    ax1.set_title('Language Distribution in Dataset', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, lang_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(val)}\n({val/lang_counts.sum()*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Pie chart
    ax2.pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%',
           colors=colors_lang, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Language Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "language_distribution.png", bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: language_distribution.png")

# ==========================================
# 3. AUDIO STATISTICS
# ==========================================
if df_audio is not None:
    print("\n3. Creating audio statistics plots...")
    
    numeric_cols = df_audio.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Duration' in numeric_cols or 'duration' in numeric_cols:
        duration_col = 'Duration' if 'Duration' in df_audio.columns else 'duration'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Duration histogram
        durations = df_audio[duration_col].dropna()
        ax1.hist(durations, bins=50, color='purple', edgecolor='darkviolet', alpha=0.7)
        ax1.set_xlabel('Duration (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Tracks', fontsize=12, fontweight='bold')
        ax1.set_title('Audio Duration Distribution', fontsize=13, fontweight='bold')
        ax1.axvline(durations.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {durations.mean():.1f}s')
        ax1.axvline(durations.median(), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {durations.median():.1f}s')
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Box plot
        ax2.boxplot(durations, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', edgecolor='navy'),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(color='navy'),
                   capprops=dict(color='navy'))
        ax2.set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Audio Duration Box Plot', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f"Min: {durations.min():.1f}s\nMax: {durations.max():.1f}s\n" + \
                    f"Mean: {durations.mean():.1f}s\nStd: {durations.std():.1f}s"
        ax2.text(1.15, durations.median(), stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / "audio_duration_statistics.png", bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: audio_duration_statistics.png")

# ==========================================
# 4. DATASET OVERVIEW
# ==========================================
print("\n4. Creating dataset overview...")

fig, ax = plt.subplots(figsize=(10, 6))

# Collect statistics
stats_data = []
if df_audio is not None:
    stats_data.append(['Audio Tracks', len(df_audio)])
if df_lyrics is not None:
    stats_data.append(['Lyrics Tracks', len(df_lyrics)])
if df_genre is not None:
    stats_data.append(['Genre-Tagged Tracks', len(df_genre)])
    if genre_cols:
        stats_data.append(['Unique Genres', len(genre_cols)])
        stats_data.append(['Avg Genres/Track', f"{df_genre[genre_cols].sum(axis=1).mean():.2f}"])

if stats_data:
    categories = [x[0] for x in stats_data]
    values = [x[1] if isinstance(x[1], (int, float)) else 0 for x in stats_data]
    
    bars = ax.barh(range(len(categories)), values, color='teal', edgecolor='darkslategray', alpha=0.7)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Overview Statistics', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val, stat) in enumerate(zip(bars, values, stats_data)):
        display_val = stat[1] if isinstance(stat[1], str) else int(val)
        ax.text(val if isinstance(val, (int, float)) else 0, i, f'  {display_val}', 
               va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_overview.png", bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: dataset_overview.png")

# ==========================================
# 5. CORRELATION ANALYSIS (if numeric features available)
# ==========================================
if df_audio is not None and len(numeric_cols) > 3:
    print("\n5. Creating correlation matrix...")
    
    # Select subset of numeric columns
    corr_cols = [col for col in numeric_cols if col != 'Music_Id'][:10]  # Max 10 for readability
    
    if corr_cols:
        corr_matrix = df_audio[corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45, ha='right')
        ax.set_yticklabels(corr_cols)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        
        # Add correlation values
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_correlation_matrix.png", bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: feature_correlation_matrix.png")

# ==========================================
# 6. SUMMARY REPORT
# ==========================================
print("\n" + "="*60)
print("GENERATING TEXT SUMMARY")
print("="*60 + "\n")

summary_lines = [
    "# Dataset EDA Summary Report\n",
    "## Generated Visualizations\n",
    ""
]

summary_lines.append("### 1. Genre Analysis")
if df_genre is not None and genre_cols:
    summary_lines.append(f"- Total unique genres: {len(genre_cols)}")
    summary_lines.append(f"- Total genre-tagged tracks: {len(df_genre)}")
    summary_lines.append(f"- Average genres per track: {df_genre[genre_cols].sum(axis=1).mean():.2f}")
    summary_lines.append(f"- Most common genre: {genre_counts.index[0]} ({int(genre_counts.values[0])} tracks)")
    summary_lines.append("")

summary_lines.append("### 2. Language Distribution")
if df_lyrics is not None and 'Language' in df_lyrics.columns:
    for lang, count in lang_counts.items():
        summary_lines.append(f"- {lang}: {count} tracks ({count/lang_counts.sum()*100:.1f}%)")
    summary_lines.append("")

summary_lines.append("### 3. Audio Statistics")
if df_audio is not None and (duration_col := 'Duration' if 'Duration' in df_audio.columns else 'duration') in df_audio.columns:
    durations = df_audio[duration_col].dropna()
    summary_lines.append(f"- Total tracks: {len(df_audio)}")
    summary_lines.append(f"- Duration range: {durations.min():.1f}s - {durations.max():.1f}s")
    summary_lines.append(f"- Average duration: {durations.mean():.1f}s ± {durations.std():.1f}s")
    summary_lines.append(f"- Median duration: {durations.median():.1f}s")
    summary_lines.append("")

summary_lines.append("\n## Visualization Files Created\n")
viz_files = sorted(output_dir.glob("*.png"))
for i, viz_file in enumerate(viz_files, 1):
    summary_lines.append(f"{i}. `{viz_file.name}`")

# Save summary
with open(output_dir / "EDA_SUMMARY.md", "w") as f:
    f.write("\n".join(summary_lines))

print("✓ Saved: EDA_SUMMARY.md")

print("\n" + "="*60)
print(f"COMPLETE! Generated {len(list(output_dir.glob('*.png')))} visualizations")
print(f"Output directory: {output_dir.absolute()}")
print("="*60 + "\n")
