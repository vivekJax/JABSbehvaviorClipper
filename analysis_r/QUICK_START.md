# R Analysis Quick Start

## Installation

Install required R packages using the installation script:

```bash
Rscript analysis_r/install_packages.R
```

This will install:
- **Bioconductor**: `rhdf5` (for HDF5 file reading)
- **CRAN**: All other required packages

**Note**: `rhdf5` must be installed via BiocManager (handled automatically by the script).

## Quick Workflow

### One Command (Recommended)

**IMPORTANT**: You must provide the correct path to your features directory!

```bash
# Run complete analysis pipeline
# Replace /path/to/your/jabs/features with your actual features directory path
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --features-dir /path/to/your/jabs/features \
  --distance-metric mahalanobis \
  --use-pca
```

**Finding your features directory:**
- Look for where your HDF5 feature files are stored
- Structure should be: `features_dir/{video_name}/{animal_id}/features.h5`
- Use absolute path if relative path doesn't work: `--features-dir /absolute/path/to/jabs/features`

This runs everything: feature extraction, clustering, visualization, videos, and outlier detection.

### Step-by-Step

```bash
# 1. Extract features
# IMPORTANT: Replace /path/to/your/jabs/features with your actual path
Rscript analysis_r/extract_bout_features.R \
  --behavior turn_left \
  --annotations-dir jabs/annotations \
  --features-dir /path/to/your/jabs/features \
  --output bout_features.csv

# 2. Cluster bouts
Rscript analysis_r/cluster_bouts.R \
  --input bout_features.csv \
  --method kmeans \
  --output-dir results/

# 3. Visualize
Rscript analysis_r/visualize_clusters.R \
  --features bout_features.csv \
  --clusters results/cluster_assignments_kmeans.csv \
  --output-dir results/

# 4. Find outliers (Mahalanobis - accounts for correlations)
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --distance-metric mahalanobis \
  --use-pca \
  --output-dir results/outlier_videos

# 5. Select bouts
Rscript analysis_r/select_bouts.R \
  --clusters results/cluster_assignments_kmeans.csv \
  --cluster-ids 0,1 \
  --output-json results/selected_bouts.json
```

## Distance Metrics

**For correlated features (recommended):**
- `--distance-metric mahalanobis` - Accounts for feature correlations
- `--use-pca` - Removes correlation via PCA reduction

**For uncorrelated features:**
- `--distance-metric euclidean` - Standard geometric distance (default)

## Bout Selection: Unfragmented Labels

The analysis uses **`unfragmented_labels`** from annotation JSON files, which contain the original bout boundaries as specified during labeling. This matches the bout counts shown in the JABS GUI.

- **`unfragmented_labels`**: Original bout start/end frames (used by analysis)
- **`labels`**: Fragmented bouts (broken up to exclude frames missing pose data)
- All bouts (both `present=True` and `present=False`) are included for comprehensive clustering and outlier detection

## Note

The Python analysis code in `analysis/` directory is preserved but not used. The R analysis in `analysis_r/` is the active analysis pipeline and does not affect the Python video clipping application.

