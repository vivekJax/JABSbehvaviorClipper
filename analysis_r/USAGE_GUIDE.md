# Usage Guide: Complete Analysis Pipeline

## Quick Start: One Command

Run the entire analysis pipeline with a single command:

```bash
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --distance-metric mahalanobis \
  --use-pca
```

This will:
1. ✅ Extract features from HDF5 files
2. ✅ Perform clustering (K-means, hierarchical, DBSCAN)
3. ✅ Create visualizations
4. ✅ Generate cluster videos
5. ✅ Find outliers using Mahalanobis distance with PCA reduction

**Output**: All results saved to `results/` directory

## Master Script Options

### Basic Usage

```bash
# IMPORTANT: Provide the correct path to your features directory
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --features-dir /path/to/your/jabs/features
```

**Note**: The default `--features-dir ../jabs/features` may not work if your features are in a different location. Always specify the correct path!

### Advanced Options

```bash
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --features-dir /path/to/your/jabs/features \
  --output-dir my_results \
  --distance-metric mahalanobis \
  --use-pca \
  --pca-variance 0.95 \
  --workers 8 \
  --verbose
```

### Skip Steps

```bash
# Skip clustering (only features and outliers)
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --skip-clustering

# Skip outlier detection
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --skip-outliers

# Skip video generation (faster, no videos)
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --skip-videos
```

## Distance Metrics for Outlier Detection

### Mahalanobis Distance (Recommended)

**Accounts for feature correlations** - statistically principled for correlated data.

```bash
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --distance-metric mahalanobis
```

**When to use**:
- Features are correlated (common in behavioral data)
- Want statistically rigorous approach
- Want to account for covariance structure

### PCA + Euclidean

**Removes correlation, reduces dimensions** - efficient for high-dimensional data.

```bash
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --distance-metric euclidean \
  --use-pca \
  --pca-variance 0.95
```

**When to use**:
- High-dimensional data (many features)
- Want dimensionality reduction
- Want computational efficiency
- Features are correlated

### Euclidean (Baseline)

**Standard geometric distance** - assumes features are uncorrelated.

```bash
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --distance-metric euclidean
```

**When to use**:
- Features are approximately uncorrelated
- Baseline for comparison
- Computational simplicity

## Complete Options Reference

```
--behavior BEHAVIOR          Behavior name (default: turn_left)
--annotations-dir DIR        Annotation directory (default: jabs/annotations)
--features-dir DIR           Feature HDF5 directory (default: ../jabs/features)
                              **REQUIRED**: Provide the actual path to your features!
                              Example: --features-dir /Users/username/data/jabs/features
--video-dir DIR              Video directory (default: .)
--output-dir DIR             Output directory (default: results)
--distance-metric METRIC     Distance metric: euclidean, mahalanobis, manhattan, cosine
                             (default: mahalanobis)
--use-pca                    Use PCA reduction for outlier detection
--pca-variance FLOAT         Variance to retain in PCA (default: 0.95)
--skip-clustering            Skip clustering analysis
--skip-outliers              Skip outlier detection
--skip-videos                Skip video generation
--workers N                  Number of parallel workers for video clipping
--verbose                    Enable verbose logging
```

## Output Structure

After running the complete analysis:

```
results/
├── bout_features.csv                    # Feature matrix
├── clustering/
│   ├── cluster_assignments_kmeans.csv
│   ├── cluster_assignments_hierarchical.csv
│   ├── cluster_statistics_*.json
│   ├── pca_clusters.png
│   ├── tsne_clusters.png
│   └── ... (other visualizations)
├── cluster_videos_kmeans/
│   ├── cluster_kmeans_1.mp4
│   ├── cluster_kmeans_2.mp4
│   └── ...
├── cluster_videos_hierarchical/
│   ├── cluster_hierarchical_1.mp4
│   └── ...
└── outlier_videos/
    ├── outliers.csv
    ├── outliers.mp4
    ├── outliers_analysis.png
    └── ... (other visualizations)
```

## Examples

### Example 1: Complete Analysis with Mahalanobis

```bash
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --distance-metric mahalanobis \
  --output-dir results_turn_left
```

### Example 2: PCA-Based Analysis

```bash
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --distance-metric euclidean \
  --use-pca \
  --pca-variance 0.90
```

### Example 3: Fast Analysis (No Videos)

```bash
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --skip-videos \
  --distance-metric mahalanobis
```

### Example 4: Only Outliers (Skip Clustering)

```bash
Rscript analysis_r/run_full_analysis.R \
  --behavior turn_left \
  --skip-clustering \
  --distance-metric mahalanobis \
  --use-pca
```

## Step-by-Step Alternative

If you prefer to run steps individually:

```bash
# 1. Extract features
Rscript analysis_r/extract_bout_features.R --behavior turn_left

# 2. Cluster
Rscript analysis_r/cluster_bouts.R --input bout_features.csv --method all

# 3. Visualize
Rscript analysis_r/visualize_clusters.R \
  --features bout_features.csv \
  --clusters cluster_assignments_kmeans.csv

# 4. Generate videos
Rscript analysis_r/generate_cluster_videos.R \
  --clusters cluster_assignments_kmeans.csv \
  --method kmeans

# 5. Find outliers
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --distance-metric mahalanobis \
  --use-pca
```

## Troubleshooting

### R Environment Issues

If you see errors about missing packages:
```bash
Rscript analysis_r/install_packages.R
```

### Memory Issues

For large datasets:
- Use `--use-pca` to reduce dimensionality
- Use `--skip-videos` to skip video generation
- Reduce `--pca-variance` (e.g., 0.90 instead of 0.95)

### Video Generation Fails

- Check video file paths in `--video-dir`
- Ensure videos exist and are accessible
- Use `--verbose` for detailed error messages

## Performance Tips

1. **Use PCA reduction**: Reduces computation time significantly
2. **Skip videos for testing**: Use `--skip-videos` for faster iteration
3. **Parallel processing**: Use `--workers N` for video generation
4. **Mahalanobis is slower**: Use PCA + Euclidean for faster outlier detection

## Next Steps

After running the analysis:

1. **Review visualizations**: Check `clustering/` directory for plots
2. **Watch cluster videos**: Validate behavioral patterns
3. **Review outliers**: Check `outlier_videos/` for unusual behaviors
4. **Select specific clusters**: Use `select_bouts.R` to export subsets

