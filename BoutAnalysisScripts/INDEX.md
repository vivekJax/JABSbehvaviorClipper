---
title: "Script Index"
description: "Quick reference index of all scripts"
category: "reference"
audience: ["users", "developers"]
tags: ["index", "reference", "scripts"]
---

# Script Index

Quick reference for all analysis scripts organized by category.

## Core Analysis Scripts

**Location**: `scripts/core/`

| Script | Description | Usage |
|-------|------------|-------|
| `extract_bout_features.R` | Extract features from HDF5 files | `Rscript scripts/core/extract_bout_features.R --behavior turn_left` |
| `extract_bout_features.py` | Python feature extraction (alternative) | `python3 scripts/core/extract_bout_features.py --behavior turn_left` |
| `cluster_bouts.R` | Clustering analysis (K-means, Hierarchical, DBSCAN) | `Rscript scripts/core/cluster_bouts.R --input features.csv --method kmeans` |
| `detect_outliers_consensus.R` | Multi-method outlier detection with consensus | `Rscript scripts/core/detect_outliers_consensus.R --features features.csv` |
| `find_outliers.R` | Single-method outlier detection | `Rscript scripts/core/find_outliers.R --features features.csv` |
| `filter_outliers.R` | Filter outliers from dataset | `Rscript scripts/core/filter_outliers.R --features features.csv` |
| `run_full_analysis.R` | Complete analysis pipeline | `Rscript scripts/core/run_full_analysis.R --behavior turn_left` |
| `select_bouts.R` | Select bouts by criteria | `Rscript scripts/core/select_bouts.R --clusters clusters.csv` |

## Visualization Scripts

**Location**: `scripts/visualization/`

| Script | Description | Usage |
|-------|------------|-------|
| `visualize_clusters.R` | Generate cluster visualizations | `Rscript scripts/visualization/visualize_clusters.R --features features.csv --clusters clusters.csv` |
| `visualize_clusters_pdf.R` | Generate PDF reports | `Rscript scripts/visualization/visualize_clusters_pdf.R --features features.csv --clusters clusters.csv` |
| `visualize_outliers.R` | Generate outlier visualizations | `Rscript scripts/visualization/visualize_outliers.R --features features.csv` |

## Video Generation Scripts

**Location**: `scripts/video/`

| Script | Description | Usage |
|-------|------------|-------|
| `generate_cluster_videos.R` | Generate videos for each cluster | `Rscript scripts/video/generate_cluster_videos.R --clusters clusters.csv --method kmeans` |
| `generate_outlier_videos.R` | Generate outlier video montages | `Rscript scripts/video/generate_outlier_videos.R --outliers outliers.csv` |

## Setup Scripts

**Location**: `setup/`

| Script | Description | Usage |
|-------|------------|-------|
| `install_packages.R` | Install R packages | `Rscript setup/install_packages.R` |
| `setup_conda.sh` | Setup Conda environment | `bash setup/setup_conda.sh` |
| `setup_venv.sh` | Setup Python venv | `bash setup/setup_venv.sh` |
| `run_full_pipeline.sh` | Shell script for complete pipeline | `bash setup/run_full_pipeline.sh` |

## Utility Functions

**Location**: `utils/`

| File | Description |
|------|-------------|
| `data_preprocessing.R` | Data preprocessing utilities |
| `visualization.R` | Visualization helper functions |

## Configuration Files

**Location**: `config/`

| File | Description |
|------|-------------|
| `requirements_R.txt` | R package requirements |
| `requirements.txt` | Python package requirements |

## Documentation

**Location**: `docs/`

| File | Description |
|------|-------------|
| `README.md` | Main documentation entry point |
| `USER_GUIDE.md` | Complete usage guide |
| `TECHNICAL_GUIDE.md` | Statistical methodology |
| `TROUBLESHOOTING.md` | Common issues and solutions |

## Workflow Examples

### Complete Pipeline
```bash
Rscript scripts/core/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features
```

### Step-by-Step
```bash
# 1. Extract features
Rscript scripts/core/extract_bout_features.R --behavior turn_left

# 2. Cluster
Rscript scripts/core/cluster_bouts.R --input bout_features.csv --method kmeans

# 3. Visualize
Rscript scripts/visualization/visualize_clusters.R \
  --features bout_features.csv \
  --clusters cluster_assignments_kmeans.csv

# 4. Detect outliers
Rscript scripts/core/detect_outliers_consensus.R \
  --features bout_features.csv \
  --distance-metric mahalanobis
```

## See Also

- [README.md](README.md) - Main entry point
- [STRUCTURE.md](STRUCTURE.md) - Directory structure details
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) - Complete usage guide

