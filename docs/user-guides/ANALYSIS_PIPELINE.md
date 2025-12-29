---
title: "Analysis Pipeline Documentation"
description: "Complete documentation of the analysis pipeline workflow"
category: "user-guide"
audience: ["users", "analysts"]
tags: ["pipeline", "workflow", "features"]
documentation_id: "analysis-pipeline"
---

# Unified Analysis Pipeline

This document describes the unified analysis pipeline that links Python bout/feature extraction with R clustering and outlier detection, with caching to save compute time.

## Overview

The pipeline consists of:
1. **Python**: Extract bouts and features from HDF5 files (with caching)
2. **R**: Cluster bouts based on features
3. **R**: Detect outliers in feature space
4. **R**: Visualize results

All steps use caching to avoid recomputation when possible.

## Quick Start

Run the complete pipeline:

```bash
python3 run_analysis_pipeline.py --behavior turn_left
```

This will:
- Extract features from HDF5 files (cached for future runs)
- Cluster bouts using K-means
- Detect outliers using distance metrics
- Generate visualizations

## Pipeline Components

### 1. Feature Extraction (Python)

**Script**: `extract_bout_features.py`

Extracts bout-level features from HDF5 files:
- Uses `unfragmented_labels` from annotation JSON files (matches GUI counts)
- Only processes `present=True` bouts
- Aggregates per-frame features to bout-level statistics (mean, std, min, max, median, etc.)
- Caches results to avoid recomputation

**Usage**:
```bash
python3 extract_bout_features.py \
    --behavior turn_left \
    --annotations-dir jabs/annotations \
    --features-dir jabs/features \
    --output BoutResults/bout_features.csv \
    --workers 7  # Optional: defaults to CPU cores - 1
```

**Caching**:
- Results are cached in `BoutResults/cache/` based on annotation file modification times
- Use `--force-recompute` to regenerate even if cache exists
- Cache key is computed from annotation files and behavior name

**Parallel Processing**:
- Default: Uses `n-1` CPU cores (leaves one core free for system responsiveness)
- Custom: Specify with `--workers N`
- Speedup: ~4-8x faster than sequential processing

**Output**: `BoutResults/bout_features.csv` with bout metadata and aggregated features

### 2. Clustering (R)

**Script**: `BoutAnalysisScripts/scripts/cluster_bouts.R`

Clusters bouts based on extracted features:
- K-means clustering (default)
- Hierarchical clustering (optional)
- DBSCAN clustering (optional)
- Distance metrics: Euclidean, Manhattan, Cosine, Mahalanobis
- Optional PCA reduction

**Usage**:
```bash
Rscript BoutAnalysisScripts/scripts/cluster_bouts.R \
    --features BoutResults/bout_features.csv \
    --output-dir BoutResults/clustering \
    --method kmeans \
    --distance-metric mahalanobis \
    --use-pca \
    --ncores 7  # Optional: defaults to CPU cores - 1
```

**Parallel Processing**:
- Default: Uses `n-1` CPU cores for optimal k search
- Custom: Specify with `--ncores N`
- Speedup: ~2-4x faster when finding optimal k

**Output**: 
- `BoutResults/clustering/cluster_assignments_kmeans.csv`
- `BoutResults/clustering/cluster_statistics_kmeans.json`
- Visualization plots

### 3. Outlier Detection (R)

**Script**: `BoutAnalysisScripts/scripts/find_outliers.R`

Detects outlier bouts using distance metrics:
- Mean/median/max distance to other bouts
- Distance metrics: Euclidean, Manhattan, Cosine, Mahalanobis
- Optional PCA reduction
- Generates outlier video montage

**Usage**:
```bash
Rscript BoutAnalysisScripts/scripts/find_outliers.R \
    --features BoutResults/bout_features.csv \
    --distance-metric mahalanobis \
    --use-pca \
    --output-dir BoutResults/outlier_videos \
    --behavior turn_left
```

**Output**:
- `BoutResults/outlier_videos/outliers.csv`
- `BoutResults/outlier_videos/outliers.mp4` (video montage)
- Visualization plots

### 4. Visualization (R)

**Script**: `BoutAnalysisScripts/scripts/visualize_clusters.R`

Creates visualizations of clustering results:
- PCA plots
- t-SNE plots
- Feature distributions
- Cluster heatmaps
- Bout timelines

**Usage**:
```bash
Rscript BoutAnalysisScripts/scripts/visualize_clusters.R \
    --features BoutResults/bout_features.csv \
    --clusters BoutResults/clustering/cluster_assignments_kmeans.csv \
    --output-dir BoutResults/clustering
```

## Unified Pipeline Script

**Script**: `run_analysis_pipeline.py`

Runs the complete pipeline in sequence:

```bash
python3 run_analysis_pipeline.py \
    --behavior turn_left \
    --annotations-dir jabs/annotations \
    --features-dir jabs/features \
    --output-dir BoutResults \
    --distance-metric mahalanobis \
    --use-pca \
    --workers 7  # Optional: defaults to CPU cores - 1 for all steps
```

**Options**:
- `--skip-features`: Skip feature extraction (use existing CSV)
- `--skip-clustering`: Skip clustering analysis
- `--skip-outliers`: Skip outlier detection
- `--skip-visualization`: Skip visualization
- `--force-recompute`: Force recomputation even if cache exists
- `--workers N`: Number of parallel workers (default: CPU cores - 1, applies to all steps)
- `--verbose`: Enable verbose logging

## Caching Strategy

### Python Feature Extraction

- **Cache location**: `results/cache/bout_features_<hash>.pkl`
- **Cache key**: Based on annotation file modification times and behavior name
- **Cache contents**: DataFrame with all features, bout metadata, timestamp
- **Cache validation**: Automatic - regenerates if annotation files change

### Benefits

1. **Speed**: Skip expensive HDF5 reads on subsequent runs
2. **Consistency**: Same cache key = same results
3. **Flexibility**: Use `--force-recompute` to regenerate when needed

## Workflow

### First Run

```bash
# Run complete pipeline (extracts features, clusters, detects outliers)
python3 run_analysis_pipeline.py --behavior turn_left
```

### Subsequent Runs (with cache)

```bash
# Features are loaded from cache automatically
# Only clustering/outlier detection runs
python3 run_analysis_pipeline.py --behavior turn_left
```

### Partial Runs

```bash
# Skip feature extraction (use existing CSV)
python3 run_analysis_pipeline.py --behavior turn_left --skip-features

# Only run clustering
python3 run_analysis_pipeline.py --behavior turn_left --skip-features --skip-outliers --skip-visualization
```

## Output Structure

All output is organized in the `BoutResults/` directory:

```
BoutResults/
├── bout_features.csv              # Feature matrix (from Python)
├── cache/                         # Python cache files
│   ├── bout_features_<hash>.pkl
│   └── bout_features_<hash>.info
├── clustering/                    # Clustering results
│   ├── cluster_assignments_kmeans.csv
│   ├── cluster_statistics_kmeans.json
│   ├── pca_clusters.png
│   ├── tsne_clusters.png
│   └── ...
└── outlier_videos/                # Outlier detection results
    ├── outliers.csv
    ├── outliers.mp4
    ├── outliers_pca.png
    └── ...
```

## Integration with Existing Scripts

The pipeline is compatible with existing R scripts:
- `BoutAnalysisScripts/scripts/extract_bout_features.R` - Checks for Python-generated CSV first
- `BoutAnalysisScripts/scripts/cluster_bouts.R` - Works with Python-generated CSV
- `BoutAnalysisScripts/scripts/find_outliers.R` - Works with Python-generated CSV
- `BoutAnalysisScripts/scripts/visualize_clusters.R` - Works with Python-generated CSV

## Performance

### With Caching and Parallel Processing

- **First run**: ~2-5 minutes (depends on number of bouts, features, and CPU cores)
- **Subsequent runs**: ~30 seconds - 1 minute (only R analysis, no HDF5 reads)

### Without Caching

- **Every run**: ~2-5 minutes (full feature extraction with parallel processing)

### Speedup from Parallel Processing

- **Feature extraction**: ~4-8x faster with n-1 cores (e.g., 8 cores = ~7x speedup)
- **Clustering**: ~2-4x faster when finding optimal k (parallel k evaluation)
- **Video generation**: Already parallelized (n-1 cores default)

## Troubleshooting

### Cache Issues

If cache seems stale or incorrect:
```bash
# Force recomputation
python3 run_analysis_pipeline.py --behavior turn_left --force-recompute
```

### Missing Features

If some bouts have no features:
- Check that HDF5 files exist in `jabs/features/<video>/<animal>/features.h5`
- Verify frame ranges are valid
- Run with `--verbose` for detailed logging

### R Script Errors

If R scripts fail:
- Ensure R packages are installed: `Rscript BoutAnalysisScripts/scripts/install_packages.R`
- Check that `BoutResults/bout_features.csv` exists
- Verify output directories exist

## Best Practices

1. **Use the unified pipeline** for complete analysis
2. **Let caching work** - don't force recompute unless needed
3. **Check cache info** in `BoutResults/cache/*.info` files
4. **Version control** - commit `BoutResults/bout_features.csv` but not cache files
5. **Clean cache** periodically: `rm -rf BoutResults/cache/*`
6. **All output** is organized in `BoutResults/` directory for cleanliness

## Notes

- All scripts use `unfragmented_labels` from annotation JSON files
- Only `present=True` bouts are included in feature extraction
- Python and R scripts produce compatible CSV formats
- Cache is automatically invalidated when annotation files change

