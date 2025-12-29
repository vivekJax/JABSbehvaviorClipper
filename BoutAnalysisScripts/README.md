---
title: "Bout Analysis Scripts"
description: "R and Python scripts for behavior bout analysis"
category: "overview"
audience: ["users", "developers"]
tags: ["analysis", "scripts", "overview"]
---

# Bout Analysis Scripts

Professional analysis pipeline for behavior bout clustering, outlier detection, and visualization.

## Quick Start

```bash
# Install dependencies
Rscript setup/install_packages.R

# Run complete analysis
Rscript scripts/core/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features
```

## Directory Structure

```
BoutAnalysisScripts/
├── scripts/
│   ├── core/              # Core analysis scripts
│   │   ├── extract_bout_features.R
│   │   ├── cluster_bouts.R
│   │   ├── detect_outliers_consensus.R
│   │   ├── find_outliers.R
│   │   ├── filter_outliers.R
│   │   ├── run_full_analysis.R
│   │   ├── select_bouts.R
│   │   └── extract_bout_features.py
│   ├── visualization/     # Visualization scripts
│   │   ├── visualize_clusters.R
│   │   ├── visualize_clusters_pdf.R
│   │   └── visualize_outliers.R
│   └── video/             # Video generation scripts
│       ├── generate_cluster_videos.R
│       └── generate_outlier_videos.R
├── docs/                  # Documentation
│   ├── README.md          # Main documentation
│   ├── USER_GUIDE.md      # Usage guide
│   ├── TECHNICAL_GUIDE.md # Technical details
│   └── TROUBLESHOOTING.md # Troubleshooting
├── utils/                 # Utility functions
├── config/                # Configuration files
│   ├── requirements_R.txt
│   └── requirements.txt
└── setup/                 # Setup scripts
    ├── install_packages.R
    ├── setup_conda.sh
    ├── setup_venv.sh
    └── run_full_pipeline.sh
```

## Scripts by Category

### Core Analysis
- **`extract_bout_features.R`** - Extract features from HDF5 files
- **`cluster_bouts.R`** - Clustering analysis (K-means, Hierarchical, DBSCAN)
- **`detect_outliers_consensus.R`** - Multi-method outlier detection
- **`find_outliers.R`** - Single-method outlier detection
- **`filter_outliers.R`** - Filter outliers from dataset
- **`run_full_analysis.R`** - Complete analysis pipeline
- **`select_bouts.R`** - Select bouts by criteria
- **`extract_bout_features.py`** - Python feature extraction (alternative)

### Visualization
- **`visualize_clusters.R`** - Cluster visualizations
- **`visualize_clusters_pdf.R`** - PDF report generation
- **`visualize_outliers.R`** - Outlier visualizations

### Video Generation
- **`generate_cluster_videos.R`** - Generate videos for each cluster
- **`generate_outlier_videos.R`** - Generate outlier videos

## Documentation

- **[docs/README.md](docs/README.md)** - Main documentation
- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** - Complete usage guide
- **[docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md)** - Statistical methodology
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues

## Installation

```bash
# Install R packages
Rscript setup/install_packages.R

# Or use conda/venv for Python
bash setup/setup_conda.sh
# or
bash setup/setup_venv.sh
```

## Usage Examples

### Complete Pipeline
```bash
Rscript scripts/core/run_full_analysis.R \
  --behavior turn_left \
  --features-dir ../jabs/features \
  --distance-metric mahalanobis \
  --use-pca
```

### Individual Steps
```bash
# Extract features
Rscript scripts/core/extract_bout_features.R --behavior turn_left

# Cluster
Rscript scripts/core/cluster_bouts.R --input bout_features.csv --method kmeans

# Detect outliers
Rscript scripts/core/detect_outliers_consensus.R --features bout_features.csv

# Visualize
Rscript scripts/visualization/visualize_clusters.R --features bout_features.csv
```

## Requirements

- R 3.6+ with packages (see `config/requirements_R.txt`)
- Python 3 (optional, for Python feature extraction)
- ffmpeg (for video generation)

## See Also

- Main project README: `../README.md`
- Analysis pipeline: `../ANALYSIS_PIPELINE.md`

