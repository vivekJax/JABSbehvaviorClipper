---
title: "Directory Structure"
description: "Professional organization of BoutAnalysisScripts"
category: "reference"
audience: ["developers", "maintainers"]
tags: ["structure", "organization", "reference"]
---

# Directory Structure

Professional organization of analysis scripts, documentation, and utilities.

## Overview

```
BoutAnalysisScripts/
├── README.md                    # Main entry point
├── STRUCTURE.md                 # This file
├── scripts/                     # All analysis scripts
│   ├── core/                    # Core analysis functionality
│   ├── visualization/           # Visualization scripts
│   └── video/                   # Video generation scripts
├── docs/                        # Documentation
├── utils/                       # Utility functions
├── config/                      # Configuration files
└── setup/                       # Setup and installation scripts
```

## Scripts Organization

### `scripts/core/` - Core Analysis

Main analysis scripts that perform the core functionality:

- **`extract_bout_features.R`** - Extract features from HDF5 files
- **`extract_bout_features.py`** - Python alternative for feature extraction
- **`cluster_bouts.R`** - Clustering analysis (K-means, Hierarchical, DBSCAN)
- **`detect_outliers_consensus.R`** - Multi-method outlier detection with consensus
- **`find_outliers.R`** - Single-method outlier detection
- **`filter_outliers.R`** - Filter outliers from dataset
- **`run_full_analysis.R`** - Complete analysis pipeline orchestrator
- **`select_bouts.R`** - Select bouts by criteria

### `scripts/visualization/` - Visualization

Scripts for creating visualizations and reports:

- **`visualize_clusters.R`** - Generate cluster visualizations (PCA, t-SNE, heatmaps)
- **`visualize_clusters_pdf.R`** - Generate multi-page PDF reports
- **`visualize_outliers.R`** - Generate outlier visualizations

### `scripts/video/` - Video Generation

Scripts for generating video montages:

- **`generate_cluster_videos.R`** - Generate videos for each cluster
- **`generate_outlier_videos.R`** - Generate outlier video montages

## Documentation

### `docs/` - Documentation

- **`README.md`** - Main documentation entry point
- **`USER_GUIDE.md`** - Complete usage guide with examples
- **`TECHNICAL_GUIDE.md`** - Statistical methodology and technical details
- **`TROUBLESHOOTING.md`** - Common issues and solutions

## Utilities

### `utils/` - Utility Functions

Shared utility functions used across scripts:

- **`data_preprocessing.R`** - Data preprocessing utilities
- **`visualization.R`** - Visualization helper functions

## Configuration

### `config/` - Configuration Files

- **`requirements_R.txt`** - R package requirements
- **`requirements.txt`** - Python package requirements

## Setup

### `setup/` - Setup Scripts

Installation and environment setup:

- **`install_packages.R`** - Install R packages
- **`setup_conda.sh`** - Conda environment setup
- **`setup_venv.sh`** - Python venv setup
- **`run_full_pipeline.sh`** - Shell script for complete pipeline

## Design Principles

1. **Separation of Concerns**: Scripts organized by function (core, visualization, video)
2. **Clear Hierarchy**: Logical grouping makes it easy to find scripts
3. **Documentation First**: Comprehensive docs in dedicated directory
4. **Reusability**: Utility functions separated for reuse
5. **Maintainability**: Clear structure makes maintenance easier

## Path Resolution

Scripts use relative paths that resolve based on script location:
- Scripts in `scripts/core/` can reference `scripts/visualization/` via `../visualization/`
- Utility functions in `utils/` are referenced via `../../utils/`
- Main pipeline scripts handle path resolution automatically

## Adding New Scripts

1. **Core functionality** → `scripts/core/`
2. **Visualization** → `scripts/visualization/`
3. **Video generation** → `scripts/video/`
4. **Utility functions** → `utils/`
5. **Documentation** → `docs/`

## Best Practices

- Keep scripts focused on a single responsibility
- Use utility functions for shared functionality
- Document scripts with clear usage examples
- Update this structure document when adding new categories

