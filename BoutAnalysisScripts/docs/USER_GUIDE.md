---
title: "User Guide - R Analysis Scripts"
description: "Complete usage guide with examples and workflows"
category: "user-guide"
audience: ["users", "analysts"]
tags: ["usage", "workflow", "examples"]
documentation_id: "r-user-guide"
---

# User Guide

Complete guide for using the R analysis scripts.

## Quick Start

### One Command (Recommended)

Run the entire pipeline:

```bash
cd JABSbehvaviorClipper
Rscript BoutAnalysisScripts/scripts/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features \
  --distance-metric mahalanobis \
  --use-pca
```

This runs: feature extraction → clustering → visualization → outlier detection → videos

**Output**: All results in `BoutResults/` directory

## Complete Pipeline Options

### Basic Usage

```bash
Rscript BoutAnalysisScripts/scripts/run_full_analysis.R \
  --behavior turn_left \
  --features-dir /path/to/jabs/features
```

**Important**: Always provide the correct path to your features directory!

### Advanced Options

```bash
Rscript BoutAnalysisScripts/scripts/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features \
  --output-dir BoutResults \
  --distance-metric mahalanobis \
  --use-pca \
  --pca-variance 0.95 \
  --workers 8 \
  --verbose
```

### Skip Steps

```bash
# Skip clustering
--skip-clustering

# Skip outlier detection
--skip-outliers

# Skip video generation (faster)
--skip-videos
```

## Distance Metrics

### Mahalanobis Distance (Recommended)

**Accounts for feature correlations** - statistically principled.

```bash
--distance-metric mahalanobis
```

**When to use**: Features are correlated (common in behavioral data)

### PCA + Euclidean

**Removes correlation, reduces dimensions** - efficient for high-dimensional data.

```bash
--distance-metric euclidean --use-pca --pca-variance 0.95
```

**When to use**: High-dimensional data, want dimensionality reduction

### Euclidean (Baseline)

**Standard geometric distance** - assumes features are uncorrelated.

```bash
--distance-metric euclidean
```

**When to use**: Features are approximately uncorrelated, baseline comparison

## Step-by-Step Workflow

### 1. Extract Features

```bash
Rscript BoutAnalysisScripts/scripts/extract_bout_features.R \
  --behavior turn_left \
  --annotations-dir ../jabs/annotations \
  --features-dir ../jabs/features \
  --output BoutResults/bout_features.csv
```

**Options**:
- `--behavior`: Behavior name (default: `turn_left`)
- `--annotations-dir`: Annotation directory (default: `../jabs/annotations`)
- `--features-dir`: Feature HDF5 directory (default: `../jabs/features`)
- `--output`: Output CSV file (default: `bout_features.csv`)
- `--verbose`: Enable verbose logging

**Output**: `bout_features.csv` - Feature matrix with bout metadata

### 2. Cluster Bouts

```bash
Rscript BoutAnalysisScripts/scripts/cluster_bouts.R \
  --features BoutResults/bout_features.csv \
  --method kmeans \
  --output-dir BoutResults/clustering
```

**Options**:
- `--features`: Input CSV with features
- `--method`: `kmeans`, `hierarchical`, `dbscan`, or `all` (default: `kmeans`)
- `--n-clusters`: Number of clusters (auto-detect if not specified)
- `--output-dir`: Output directory (default: current directory)
- `--ncores`: Number of parallel cores (default: CPU cores - 1)

**Output**:
- `cluster_assignments_{method}.csv`: Cluster assignments
- `cluster_statistics_{method}.json`: Cluster statistics

### 3. Visualize Clusters

```bash
Rscript BoutAnalysisScripts/scripts/visualize_clusters.R \
  --features BoutResults/bout_features.csv \
  --clusters BoutResults/clustering/cluster_assignments_kmeans.csv \
  --output-dir BoutResults/clustering
```

**Output**:
- `pca_clusters.png`: PCA 2D scatter plot
- `tsne_clusters.png`: t-SNE 2D scatter plot
- `feature_distributions.png`: Feature distributions by cluster
- `cluster_heatmap.png`: Heatmap of feature means
- `cluster_sizes.png`: Cluster size distribution

### 4. Generate PDF Reports

```bash
Rscript BoutAnalysisScripts/scripts/visualize_clusters_pdf.R \
  --features BoutResults/bout_features.csv \
  --clusters BoutResults/clustering/cluster_assignments_kmeans.csv \
  --output-dir BoutResults/clustering
```

**Output**: `clustering_kmeans_report.pdf` - Multi-page PDF with all visualizations

### 5. Detect Outliers

```bash
Rscript BoutAnalysisScripts/scripts/detect_outliers_consensus.R \
  --features BoutResults/bout_features.csv \
  --output-dir BoutResults/outliers \
  --distance-metric mahalanobis \
  --use-pca \
  --pca-variance 0.95
```

**Options**:
- `--features`: Input CSV with features
- `--distance-metric`: `euclidean`, `mahalanobis`, `manhattan`, `cosine` (default: `euclidean`)
- `--use-pca`: Use PCA reduction before distance calculation
- `--pca-variance`: Proportion of variance to retain (default: `0.95`)
- `--threshold`: Outlier threshold: `auto` (top 5%), `topN`, or percentile (default: `auto`)
- `--top-n`: Number of top outliers (used with `--threshold topN`)
- `--consensus-min`: Minimum methods needed for consensus (default: `2`)

**Output**:
- `consensus_outliers.csv`: Consensus outliers
- `outlier_scores.csv`: Outlier scores from all methods
- `outlier_analysis.pdf`: Outlier analysis visualizations

### 6. Generate Cluster Videos

```bash
Rscript BoutAnalysisScripts/scripts/generate_cluster_videos.R \
  --clusters BoutResults/clustering/cluster_assignments_kmeans.csv \
  --method kmeans \
  --output-dir BoutResults/videos
```

**Output**: Videos for each cluster in `BoutResults/videos/kmeans/`

## Examples

### Example 1: Complete Analysis with Mahalanobis

```bash
Rscript BoutAnalysisScripts/scripts/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features \
  --distance-metric mahalanobis \
  --output-dir BoutResults
```

### Example 2: PCA-Based Analysis

```bash
Rscript BoutAnalysisScripts/scripts/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features \
  --distance-metric euclidean \
  --use-pca \
  --pca-variance 0.90
```

### Example 3: Fast Analysis (No Videos)

```bash
Rscript BoutAnalysisScripts/scripts/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features \
  --skip-videos \
  --distance-metric mahalanobis
```

### Example 4: Only Outliers (Skip Clustering)

```bash
Rscript BoutAnalysisScripts/scripts/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features \
  --skip-clustering \
  --distance-metric mahalanobis \
  --use-pca
```

## Output Structure

```
BoutResults/
├── bout_features.csv              # Feature matrix
├── clustering/
│   ├── cluster_assignments_kmeans.csv
│   ├── cluster_statistics_kmeans.json
│   ├── clustering_kmeans_report.pdf
│   └── videos/                    # Cluster videos
├── outliers/
│   ├── consensus_outliers.csv
│   ├── outlier_scores.csv
│   └── outlier_analysis.pdf
└── videos/
    ├── all_bouts.mp4
    ├── outliers.mp4
    └── kmeans/                    # Cluster videos
```

## Performance Tips

1. **Use PCA reduction**: Reduces computation time significantly
2. **Skip videos for testing**: Use `--skip-videos` for faster iteration
3. **Parallel processing**: Use `--ncores N` for clustering
4. **Mahalanobis is slower**: Use PCA + Euclidean for faster outlier detection

## Next Steps

After running the analysis:

1. **Review visualizations**: Check `clustering/` directory for plots
2. **Read PDF reports**: Comprehensive analysis in PDF format
3. **Watch cluster videos**: Validate behavioral patterns
4. **Review outliers**: Check `outliers/` for unusual behaviors
5. **Select specific clusters**: Use `select_bouts.R` to export subsets

## See Also

- **[README.md](README.md)** - Installation and quick start
- **[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)** - Statistical methodology and technical details
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

