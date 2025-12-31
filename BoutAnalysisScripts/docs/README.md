---
title: "R Analysis Scripts - README"
description: "Main entry point for R-based behavior bout analysis"
category: "user-guide"
audience: ["users", "analysts"]
tags: ["r", "analysis", "getting-started"]
documentation_id: "r-readme"
---

# Bout Analysis Scripts

R and Python analysis pipeline for clustering behavior bouts and detecting outliers.

## Quick Start

### 1. Install Dependencies

```bash
Rscript setup/install_packages.R
```

### 2. Run Complete Pipeline

```bash
cd JABSbehvaviorClipper
Rscript BoutAnalysisScripts/scripts/core/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features
```

This runs: feature extraction → clustering → visualization → outlier detection → videos

**Output**: All results in `BoutResults/` directory

## What This Does

1. **Feature Extraction**: Extracts per-frame features from HDF5 files and aggregates to bout-level statistics
2. **Clustering**: Groups similar bouts using K-means, hierarchical, or DBSCAN
3. **Visualization**: Creates PCA plots, t-SNE plots, heatmaps, and feature distributions
4. **Outlier Detection**: Identifies unusual bouts using distance-based methods
5. **Video Generation**: Creates videos of clusters and outliers

## Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete usage guide with examples
- **[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)** - Statistical methodology and technical details
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[TRAJECTORY_PLOTTING.md](TRAJECTORY_PLOTTING.md)** - Trajectory plotting: centroid calculation and visualization

## Installation

### Automatic (Recommended)

```bash
Rscript setup/install_packages.R
```

### Manual

```r
# Install BiocManager
install.packages("BiocManager", repos = "https://cran.r-project.org")

# Install Bioconductor package (rhdf5)
BiocManager::install("rhdf5", ask = FALSE)

# Install CRAN packages
install.packages(c("optparse", "dplyr", "jsonlite", "ggplot2", 
                   "Rtsne", "factoextra", "pheatmap", "cluster", 
                   "NbClust", "gridExtra", "dbscan", "parallel", "MASS"), 
                 repos = "https://cran.r-project.org")
```

**Note**: `rhdf5` must be installed via BiocManager (not directly from CRAN).

## Basic Usage

### Extract Features

```bash
Rscript scripts/core/extract_bout_features.R \
  --behavior turn_left \
  --annotations-dir ../jabs/annotations \
  --features-dir ../jabs/features \
  --output BoutResults/bout_features.csv
```

### Cluster Bouts

```bash
Rscript scripts/core/cluster_bouts.R \
  --input BoutResults/bout_features.csv \
  --method kmeans \
  --output-dir BoutResults/clustering
```

### Detect Outliers

```bash
Rscript scripts/core/detect_outliers_consensus.R \
  --features BoutResults/bout_features.csv \
  --output-dir BoutResults/outliers \
  --distance-metric mahalanobis \
  --use-pca
```

## Key Features

- ✅ Multiple clustering methods (K-means, Hierarchical, DBSCAN)
- ✅ Multi-method outlier detection with consensus
- ✅ Comprehensive visualizations (PCA, t-SNE, heatmaps)
- ✅ Video generation for clusters and outliers
- ✅ Parallel processing support
- ✅ Uses `unfragmented_labels` to match GUI counts
- ✅ Only processes `present=True` bouts

## Output Structure

```
BoutResults/
├── bout_features.csv              # Feature matrix
├── clustering/
│   ├── cluster_assignments_kmeans.csv
│   ├── clustering_kmeans_report.pdf
│   └── videos/                    # Cluster videos
├── outliers/
│   ├── consensus_outliers.csv
│   └── outliers.mp4
└── cache/                          # Cached results
```

## Requirements

- R 3.6+ with required packages (see Installation)
- Python 3 (for feature extraction, optional)
- ffmpeg (for video generation)
- h5dump (for HDF5 inspection, optional)

## Script Organization

- **`scripts/core/`** - Core analysis scripts
- **`scripts/visualization/`** - Visualization scripts
- **`scripts/video/`** - Video generation scripts
- **`utils/`** - Utility functions
- **`config/`** - Configuration files
- **`setup/`** - Setup scripts

See [../STRUCTURE.md](../STRUCTURE.md) for complete directory structure.

## Next Steps

- See [USER_GUIDE.md](USER_GUIDE.md) for detailed usage and examples
- See [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) for statistical methodology
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you encounter issues
