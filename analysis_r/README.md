# Behavior Bout Clustering Analysis (R Version)

This R-based analysis project clusters behavior bouts (e.g., `turn_left`) based on features extracted from JABS HDF5 files. It enables you to identify distinct behavioral patterns and select specific bouts for further analysis or video generation.

**Note:** This R analysis is completely separate from the Python video clipping application (`generate_bouts_video.py`), which remains unchanged.

## Documentation

For detailed statistical and methodological explanations, see:

- **[STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md)**: Comprehensive statistical explanations of all methods, assumptions, and rationale
- **[ANALYSIS_PIPELINE.md](ANALYSIS_PIPELINE.md)**: Complete pipeline overview with step-by-step explanations
- **[OUTLIER_EXPLANATION.md](OUTLIER_EXPLANATION.md)**: Detailed explanation of outlier detection and visualizations
- **[QUICK_START.md](QUICK_START.md)**: Quick reference for common commands

## Overview

The analysis pipeline consists of four main steps:

1. **Feature Extraction**: Extract per-frame features from HDF5 files for each behavior bout
2. **Clustering**: Perform clustering analysis using multiple algorithms (K-means, hierarchical, DBSCAN)
3. **Visualization**: Create comprehensive visualizations of clusters and feature distributions
4. **Bout Selection**: Select specific bouts based on cluster membership or feature criteria

## Installation

### Automatic Installation (Recommended)

Run the installation script:

```bash
Rscript analysis_r/install_packages.R
```

This will install all required packages including:
- **Bioconductor packages**: `rhdf5` (for HDF5 file reading)
- **CRAN packages**: `optparse`, `dplyr`, `jsonlite`, `ggplot2`, `Rtsne`, `factoextra`, `pheatmap`, `cluster`, `NbClust`, `gridExtra`, `dbscan`

### Manual Installation

If the automatic installation doesn't work, install packages manually:

```r
# Install BiocManager first
install.packages("BiocManager", repos = "https://cran.r-project.org")

# Install Bioconductor package (rhdf5)
BiocManager::install("rhdf5", ask = FALSE)

# Install CRAN packages
install.packages(c("getopt", "optparse", "dplyr", "jsonlite", "ggplot2", 
                   "Rtsne", "factoextra", "pheatmap", "cluster", "NbClust", 
                   "gridExtra", "dbscan"), 
                 repos = "https://cran.r-project.org")
```

**Note**: `rhdf5` is a Bioconductor package and must be installed via `BiocManager`, not directly from CRAN.

## Usage

### Step 1: Extract Features

Extract features for all behavior bouts:

```bash
Rscript analysis_r/extract_bout_features.R \
  --behavior turn_left \
  --annotations-dir jabs/annotations \
  --features-dir ../jabs/features \
  --output bout_features.csv
```

**Options:**
- `--behavior`: Behavior name (default: `turn_left`)
- `--annotations-dir`: Directory with annotation JSON files (default: `jabs/annotations`)
- `--features-dir`: Base directory for feature HDF5 files (default: `../jabs/features`)
- `--output`: Output CSV file (default: `bout_features.csv`)
- `--verbose`: Enable verbose logging

**Note:** The script uses `unfragmented_labels` from annotation files (original bout boundaries) to match GUI counts. All bouts (both `present=True` and `present=False`) are included for analysis.

**Output:**
- `bout_features.csv`: Feature matrix with bout metadata and aggregated features (includes `present` column indicating behavior label)

### Step 2: Cluster Bouts

Perform clustering analysis:

```bash
Rscript analysis_r/cluster_bouts.R \
  --input bout_features.csv \
  --method kmeans \
  --output-dir results/
```

**Options:**
- `--input`: Input CSV with features (default: `bout_features.csv`)
- `--method`: Clustering method: `kmeans`, `hierarchical`, `dbscan`, or `all` (default: `kmeans`)
- `--n-clusters`: Number of clusters (auto-detect if not specified)
- `--scale-method`: Feature scaling: `standard`, `minmax`, or `robust` (default: `standard`)
- `--output-dir`: Output directory (default: current directory)
- `--verbose`: Enable verbose logging

**Output:**
- `cluster_assignments_{method}.csv`: Cluster assignments for each bout
- `cluster_statistics_{method}.json`: Cluster statistics and metrics

### Step 3: Visualize Clusters

Create visualizations:

```bash
Rscript analysis_r/visualize_clusters.R \
  --features bout_features.csv \
  --clusters cluster_assignments_kmeans.csv \
  --output-dir results/
```

**Options:**
- `--features`: Input CSV with features
- `--clusters`: Input CSV with cluster assignments
- `--output-dir`: Output directory for plots
- `--verbose`: Enable verbose logging

**Output:**
- `pca_clusters.png`: PCA 2D scatter plot
- `tsne_clusters.png`: t-SNE 2D scatter plot
- `feature_distributions.png`: Feature distributions by cluster
- `cluster_heatmap.png`: Heatmap of feature means per cluster
- `cluster_sizes.png`: Cluster size distribution
- `bout_timeline.png`: Timeline of bouts colored by cluster

### Step 4: Find Outliers

Identify unusual behavior bouts using distance-based methods:

```bash
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --output-dir results/outlier_videos
```

**Options:**
- `--features`: Input CSV with features (default: `bout_features.csv`)
- `--method`: Outlier detection method: `mean_distance`, `median_distance`, `max_distance`, or `knn_distance` (default: `mean_distance`)
- `--threshold`: Outlier threshold: `auto` (top 5%), `topN`, or numeric percentile (default: `auto`)
- `--top-n`: Number of top outliers (used with `--threshold topN`)
- `--distance-metric`: Distance metric: `euclidean`, `manhattan`, `cosine`, or `mahalanobis` (default: `euclidean`)
- `--use-pca`: Use PCA-reduced dimensions for distance calculation (removes correlation)
- `--pca-variance`: Proportion of variance to retain in PCA (default: `0.95`)
- `--scale-method`: Feature scaling: `standard`, `minmax`, or `robust` (default: `standard`)
- `--output-dir`: Output directory (default: `outlier_videos`)
- `--video-dir`: Directory containing video files (default: `.`)
- `--verbose`: Enable verbose logging

**Distance Metrics Explained:**
- **Euclidean** (default): Standard geometric distance. Assumes features are uncorrelated.
- **Mahalanobis**: Accounts for feature correlations via covariance structure. **Recommended for correlated features.**
- **Manhattan**: L1 norm, more robust to outliers.
- **Cosine**: Scale-invariant, direction-based.

**PCA Reduction:**
- Use `--use-pca` to calculate distances on PCA-reduced dimensions
- Removes correlation (components are uncorrelated)
- Reduces dimensionality (e.g., 183 → 43 features, 95% variance retained)
- **Recommended for high-dimensional, correlated data**

**Output:**
- `outliers.csv`: Outlier bout information with aggregate distances
- `outliers.mp4`: Video montage of outlier bouts
- `outliers_analysis.png`: Combined analysis plots
- `outliers_pca.png`: PCA visualization
- `outliers_tsne.png`: t-SNE visualization
- `outliers_features.png`: Feature comparison plot
- `outliers_summary.png`: Summary statistics

**Examples:**
```bash
# Default (Euclidean distance, top 5%)
Rscript analysis_r/find_outliers.R --features bout_features.csv

# Mahalanobis distance (accounts for correlations)
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --distance-metric mahalanobis

# PCA reduction + Euclidean (removes correlation, reduces dimensions)
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --use-pca \
  --pca-variance 0.95

# Top 10 outliers using Mahalanobis
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --distance-metric mahalanobis \
  --threshold topN \
  --top-n 10
```

### Step 5: Select Bouts

Select bouts based on cluster analysis:

```bash
Rscript analysis_r/select_bouts.R \
  --clusters cluster_assignments_kmeans.csv \
  --cluster-ids 0,1 \
  --output-json selected_bouts.json
```

**Options:**
- `--clusters`: Input CSV with cluster assignments (required)
- `--features`: Optional: Input CSV with features for filtering
- `--cluster-ids`: Select bouts from these cluster IDs (comma-separated)
- `--animal-ids`: Select bouts from these animal IDs (comma-separated)
- `--videos`: Select bouts from these video names (comma-separated)
- `--feature-filter`: Filter by feature range: `FEATURE_NAME,MIN,MAX`
- `--output-json`: Output JSON file for selected bouts
- `--output-csv`: Output CSV file for selected bouts
- `--verbose`: Enable verbose logging

**Output:**
- `selected_bouts.json`: JSON file with selected bout metadata (compatible with video clipper)
- `selected_bouts.csv`: CSV file with selected bouts

## Complete Workflow Example

### Option 1: Run Everything with One Command (Recommended)

```bash
# Run complete analysis pipeline
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --output-dir results \
  --distance-metric mahalanobis \
  --use-pca
```

This single command will:
1. Extract features from HDF5 files
2. Perform clustering (K-means, hierarchical, DBSCAN)
3. Create visualizations
4. Generate cluster videos
5. Find outliers using Mahalanobis distance with PCA reduction

### Option 2: Step-by-Step Workflow

```bash
# 1. Extract features
Rscript analysis_r/extract_bout_features.R \
  --behavior turn_left \
  --annotations-dir jabs/annotations \
  --features-dir ../jabs/features \
  --output bout_features.csv

# 2. Cluster bouts
Rscript analysis_r/cluster_bouts.R \
  --input bout_features.csv \
  --method all \
  --output-dir results/clustering/

# 3. Visualize
Rscript analysis_r/visualize_clusters.R \
  --features bout_features.csv \
  --clusters results/clustering/cluster_assignments_kmeans.csv \
  --output-dir results/clustering/

# 4. Generate cluster videos
Rscript analysis_r/generate_cluster_videos.R \
  --clusters results/clustering/cluster_assignments_kmeans.csv \
  --method kmeans \
  --output-dir results/cluster_videos_kmeans

# 5. Find outliers (with Mahalanobis distance)
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --distance-metric mahalanobis \
  --use-pca \
  --output-dir results/outlier_videos

# 6. Select specific clusters (optional)
Rscript analysis_r/select_bouts.R \
  --clusters results/clustering/cluster_assignments_kmeans.csv \
  --cluster-ids 0,1 \
  --output-json results/selected_bouts.json
```

## Feature Extraction Details

### Bout Selection: Unfragmented Labels

The analysis uses **`unfragmented_labels`** from annotation JSON files, which contain the original bout boundaries as specified during labeling. This matches the bout counts shown in the JABS GUI.

**Important Notes:**
- **`unfragmented_labels`**: Original bout start/end frames (used by analysis)
- **`labels`**: Fragmented bouts (broken up to exclude frames missing pose data)
- The analysis includes **all bouts** (both `present=True` and `present=False`) for comprehensive clustering and outlier detection
- If `unfragmented_labels` is not found, the script falls back to `labels` with a warning

### Feature Aggregation

Per-frame features are aggregated to bout-level using:
- **Mean**: Average value across frames
- **Std**: Standard deviation
- **Min/Max**: Minimum and maximum values
- **Median**: Median value
- **First/Last**: First and last frame values
- **Duration**: Number of frames in bout

### Available Features

Features are extracted from `/features/per_frame/` group in HDF5 files, including:
- **Angles**: Body part angles (BASE_NECK-CENTER_SPINE-BASE_TAIL, etc.)
- **Angular velocity**: Rate of angle change
- **Centroid velocity**: Speed and direction of movement
- **Distances**: Pairwise distances between body parts
- **Social distances**: Distances to other animals
- **Lixit distances**: Distances to lixit (water source)
- **And more**: See HDF5 file structure for complete list

## Clustering Methods

### K-means
- **Best for**: Well-separated, spherical clusters
- **Auto-detection**: Uses elbow method and silhouette score
- **Parameters**: Number of clusters (k)

### Hierarchical (Agglomerative)
- **Best for**: Hierarchical structure, varying cluster sizes
- **Linkage**: Ward (default), complete, or average
- **Parameters**: Number of clusters

### DBSCAN
- **Best for**: Irregular shapes, noise detection
- **Parameters**: eps (distance threshold), minPts
- **Note**: May label some points as noise (0)

## Visualization Guide

### PCA Plot
- Shows clusters in 2D space using principal components
- Explained variance shows how much information is retained
- Useful for understanding overall cluster separation

### t-SNE Plot
- Non-linear dimensionality reduction
- Better at preserving local structure
- Useful for identifying subtle cluster patterns

### Feature Distributions
- Violin plots show feature value distributions per cluster
- Helps identify which features distinguish clusters
- Top features by variance are automatically selected

### Cluster Heatmap
- Shows mean feature values per cluster
- Top 20 features by variance across clusters
- Useful for understanding cluster characteristics

### Bout Timeline
- Shows when bouts occur in each video
- Colored by cluster
- Useful for temporal pattern analysis

## Integration with Video Clipper

Selected bouts can be used with the Python video clipper:

1. Export selected bouts to JSON:
   ```bash
   Rscript analysis_r/select_bouts.R \
     --clusters results/cluster_assignments_kmeans.csv \
     --cluster-ids 0 \
     --output-json selected_cluster_0.json
   ```

2. The JSON format is compatible with the video clipper's expected format

## Troubleshooting

### "No features extracted" Error

If you see `Successfully extracted features for 0/N bouts`, this means feature files aren't being found.

**Common causes**:
1. Feature directory path is incorrect (`--features-dir`)
2. Feature file structure doesn't match expected pattern
3. Video names or animal IDs don't match between annotations and features

**Solutions**:
1. **Run with verbose mode** to see detailed diagnostics:
   ```bash
   Rscript analysis_r/run_full_analysis.R --behavior turn_left --verbose
   ```

2. **Check feature directory structure**:
   - Expected: `features_dir/{video_basename}/{animal_id}/features.h5`
   - Verify paths match between annotations and feature files

3. **Use absolute paths** if relative paths are unclear:
   ```bash
   Rscript analysis_r/run_full_analysis.R \
     --features-dir /absolute/path/to/features
   ```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting guide.

### HDF5 file reading errors
- **Check**: `rhdf5` package is installed: `install.packages("rhdf5")`
- **Check**: Feature file paths match video names and animal IDs
- **Solution**: Verify `jabs/features/{video_name}/{animal_id}/features.h5` exists
- **Run with --verbose**: Shows detailed path information and HDF5 structure

### Clustering fails
- **Check**: Sufficient samples (need at least 2 samples per cluster)
- **Check**: Features have sufficient variance
- **Solution**: Try different scaling methods or reduce number of clusters

### Visualizations are empty
- **Check**: Cluster assignments file matches features file
- **Check**: Both files have matching `bout_id` values
- **Solution**: Re-run clustering with matching input files

## R Package Dependencies

Required packages:
- `rhdf5`: HDF5 file reading
- `dplyr`: Data manipulation
- `jsonlite`: JSON file reading/writing
- `optparse`: Command-line argument parsing
- `ggplot2`: Plotting
- `Rtsne`: t-SNE dimensionality reduction
- `factoextra`: Cluster analysis and visualization
- `pheatmap`: Heatmap creation
- `cluster`: Clustering algorithms
- `gridExtra`: Plot arrangement

Optional:
- `dbscan`: DBSCAN clustering
- `NbClust`: Optimal cluster number determination

## Output Files Reference

- `bout_features.csv`: Feature matrix (bouts × features)
- `cluster_assignments_{method}.csv`: Cluster IDs for each bout
- `cluster_statistics_{method}.json`: Cluster metrics and statistics
- `selected_bouts.json`: Selected bout metadata (JSON format)
- `selected_bouts.csv`: Selected bout data (CSV format)
- Visualization PNG files

## Differences from Python Version

- Uses R's `rhdf5` package instead of Python's `h5py`
- Uses R's `ggplot2` for visualizations instead of matplotlib/seaborn
- Uses R's clustering packages (`cluster`, `factoextra`) instead of scikit-learn
- Command-line interface uses `optparse` instead of `argparse`
- All scripts are R scripts (`.R`) instead of Python scripts (`.py`)

