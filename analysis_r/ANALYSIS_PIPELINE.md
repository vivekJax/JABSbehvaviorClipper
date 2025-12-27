# Complete Analysis Pipeline Documentation

## Overview

This document provides a step-by-step guide to the complete behavior bout analysis pipeline, explaining what each script does, why it's necessary, and how the pieces fit together.

## Pipeline Architecture

```
Feature Extraction → Clustering → Visualization → Outlier Detection → Video Generation
```

Each step builds on the previous, creating a comprehensive analysis workflow.

---

## Step 1: Feature Extraction (`extract_bout_features.R`)

### What It Does

Extracts per-frame features from HDF5 files and aggregates them to bout-level statistics.

### Why This Step Is Necessary

**Problem**: 
- Behavior bouts are temporal sequences (time series)
- Each frame has multiple features (angles, velocities, distances, etc.)
- Statistical methods need fixed-length feature vectors

**Solution**:
- Aggregate per-frame features into bout-level statistics
- Creates comparable feature vectors for all bouts

### Statistical Rationale

**Aggregation Strategy**:
- **Mean**: Captures typical behavior level
- **Std**: Captures variability/consistency
- **Min/Max**: Captures extreme values
- **Median**: Robust central tendency
- **First/Last**: Captures temporal dynamics
- **Duration**: Bout length as a feature

**Why Multiple Statistics?**
- Different statistics capture different aspects of behavior
- Mean alone loses information about variability
- Combining statistics provides richer feature representation

### Output

- `bout_features.csv`: Feature matrix (bouts × features)
  - Each row = one bout
  - Each column = one aggregated feature
  - Includes metadata (video, animal, frames, etc.)

### Key Decisions

1. **Which features to extract?**
   - **Decision**: Extract all available features
   - **Rationale**: Let data-driven methods identify important features
   - **Alternative**: Manual feature selection (risks losing information)

2. **Which statistics to compute?**
   - **Decision**: Mean, std, min, max, median, first, last, duration
   - **Rationale**: Comprehensive coverage of distribution and dynamics
   - **Trade-off**: More features = more information, but higher dimensionality

---

## Step 2: Clustering Analysis (`cluster_bouts.R`)

### What It Does

Groups similar behavior bouts into clusters using unsupervised learning.

### Why This Step Is Necessary

**Problem**:
- Need to discover behavioral subtypes
- Understand behavioral diversity
- Guide further analysis

**Solution**:
- Apply clustering algorithms to identify groups
- Use multiple algorithms for robustness

### Statistical Methods

#### K-Means Clustering

**Algorithm**:
1. Initialize k cluster centers randomly
2. Assign each point to nearest center
3. Update centers to cluster means
4. Repeat until convergence

**Why K-Means?**
- **Fast**: O(nkd) complexity
- **Interpretable**: Clear cluster centroids
- **Standard**: Most widely used
- **Scalable**: Works with many samples

**Optimal k Selection**:
- **Method**: Silhouette score maximization
- **Rationale**: Balances within-cluster cohesion and between-cluster separation
- **Process**: Test k=2 to k=10, select k with highest silhouette

#### Hierarchical Clustering

**Algorithm**:
1. Start with each point as its own cluster
2. Find two closest clusters
3. Merge them
4. Repeat until desired number of clusters

**Why Hierarchical?**
- **No k required**: Can examine dendrogram
- **Flexible**: Handles non-spherical clusters
- **Complementary**: Different assumptions than K-means

**Linkage Method**: Ward's
- **Why Ward?**: Minimizes within-cluster variance (similar to K-means objective)
- **Result**: Produces compact, spherical clusters

#### DBSCAN

**Algorithm**:
1. For each point, find neighbors within eps distance
2. If point has ≥ minPts neighbors, start a cluster
3. Expand cluster by adding neighbors of cluster points
4. Points not in any cluster are noise

**Why DBSCAN?**
- **Automatic k**: Discovers number of clusters
- **Noise detection**: Identifies outliers
- **Flexible shapes**: Can find non-spherical clusters

**Limitations**:
- Parameter sensitive (eps, minPts)
- May struggle with varying densities

### Validation Metrics

**Silhouette Score**:
- Measures how well each point fits its cluster
- Range: [-1, 1], higher is better
- **Why use**: Standard internal validation metric

**Calinski-Harabasz Index**:
- Ratio of between-cluster to within-cluster variance
- Higher is better
- **Why use**: Complementary to silhouette, focuses on overall structure

### Output

- `cluster_assignments_{method}.csv`: Cluster ID for each bout
- `cluster_statistics_{method}.json`: Cluster quality metrics and statistics

### Key Decisions

1. **Which algorithms to use?**
   - **Decision**: K-means, hierarchical, DBSCAN
   - **Rationale**: Different assumptions, complementary perspectives
   - **Result**: More robust conclusions

2. **How to select k?**
   - **Decision**: Silhouette score maximization
   - **Rationale**: Standard, interpretable, balances cohesion and separation
   - **Alternative**: Elbow method (more subjective)

---

## Step 3: Visualization (`visualize_clusters.R`)

### What It Does

Creates visualizations to understand cluster structure and validate results.

### Why This Step Is Necessary

**Problem**:
- Clustering results need validation
- Need to understand cluster characteristics
- Need to communicate results

**Solution**:
- Multiple visualization types
- Dimensionality reduction for 2D visualization
- Feature analysis plots

### Visualization Types

#### 1. PCA Plot

**What**: 2D projection using Principal Component Analysis

**Why**:
- **Dimensionality reduction**: Visualize high-dimensional data
- **Variance preservation**: First 2 PCs capture most information
- **Cluster validation**: See if clusters are separated in feature space

**Interpretation**:
- **Separated clusters**: Good clustering
- **Overlapping clusters**: May indicate poor clustering or similar behaviors
- **Outliers**: Points far from main groups

#### 2. t-SNE Plot

**What**: Non-linear 2D projection using t-SNE

**Why**:
- **Non-linear relationships**: Captures complex patterns PCA might miss
- **Local structure**: Preserves neighborhoods well
- **Complementary**: Different perspective from PCA

**Interpretation**:
- **Clusters**: Groups of points
- **Separation**: Clear boundaries indicate distinct behaviors
- **Manifolds**: Curved structures may indicate continuous variation

#### 3. Feature Distributions

**What**: Violin plots showing feature distributions per cluster

**Why**:
- **Feature importance**: Identify which features distinguish clusters
- **Distribution shape**: Understand feature characteristics
- **Interpretability**: Connect statistical differences to behavior

**Interpretation**:
- **Different distributions**: Features that distinguish clusters
- **Similar distributions**: Features that don't vary across clusters
- **Skewness**: May indicate non-normal distributions

#### 4. Cluster Heatmap

**What**: Heatmap of mean feature values per cluster

**Why**:
- **Overview**: See all features at once
- **Patterns**: Identify feature patterns across clusters
- **Comparison**: Easy to compare clusters

**Interpretation**:
- **Color intensity**: Magnitude of feature value
- **Patterns**: Similar colors indicate similar feature values
- **Clusters**: Rows show cluster characteristics

### Output

- `pca_clusters.png`: PCA visualization
- `tsne_clusters.png`: t-SNE visualization
- `feature_distributions.png`: Feature distributions by cluster
- `cluster_heatmap.png`: Feature means heatmap
- `cluster_sizes.png`: Cluster size distribution
- `bout_timeline.png`: Temporal distribution of bouts

### Key Decisions

1. **Which dimensionality reduction?**
   - **Decision**: Both PCA and t-SNE
   - **Rationale**: Linear (PCA) and non-linear (t-SNE) perspectives
   - **Result**: More comprehensive understanding

2. **Which features to plot?**
   - **Decision**: Top features by variance
   - **Rationale**: Most informative features
   - **Trade-off**: May miss important low-variance features

---

## Step 4: Outlier Detection (`find_outliers.R`)

### What It Does

Identifies behavior bouts that are unlike the majority using distance-based methods.

### Why This Step Is Necessary

**Problem**:
- Clustering finds groups, but what about unusual individual bouts?
- Some behaviors may be rare but important
- Need to identify atypical patterns

**Solution**:
- Compute distances between all bouts
- Identify bouts with high aggregate distances
- Generate visualizations explaining why they're outliers

### Statistical Methods

#### Distance Calculation

**Euclidean Distance** (default):
```
d(x, y) = √Σ(xi - yi)²
```

**Why Euclidean?**
- Standard, interpretable
- Works well after scaling
- Fast computation

**After Scaling**: Features are on same scale, so Euclidean is appropriate

#### Aggregate Distance Metrics

**Mean Distance** (default):
- Average distance to all other bouts
- **Why**: Comprehensive, stable, interpretable

**Median Distance**:
- Median distance to all other bouts
- **Why**: Robust to outliers

**Max Distance**:
- Maximum distance to any other bout
- **Why**: Identifies extreme outliers

**KNN Distance**:
- Average distance to k nearest neighbors
- **Why**: Focuses on local neighborhood

#### Outlier Selection

**Top 5%** (default):
- Select bouts with highest 5% of distances
- **Why**: Standard threshold, balances sensitivity and specificity

**Top N**:
- Select exactly N most distant bouts
- **Why**: When you need a specific number

**Percentile Threshold**:
- Custom percentile (e.g., 0.9 = top 10%)
- **Why**: Flexible, adapts to data

### Validation Visualizations

**Distance Distribution**:
- Shows where outliers fall in distribution
- **Why**: Visual confirmation of threshold

**Feature Space Plots**:
- Shows outliers separated from main cluster
- **Why**: Spatial validation of outlier status

**Feature Comparison**:
- Shows which features make outliers different
- **Why**: Interpretability, understand differences

### Output

- `outliers.csv`: Outlier bout information with distances
- `outliers.mp4`: Video of outlier bouts
- Multiple visualization PNGs explaining outliers

### Key Decisions

1. **Which distance metric?**
   - **Decision**: Euclidean (default), with options for Manhattan and cosine
   - **Rationale**: Standard, works well after scaling
   - **Alternative**: Cosine for high-dimensional data

2. **Which aggregate metric?**
   - **Decision**: Mean distance (default)
   - **Rationale**: Comprehensive, stable, interpretable
   - **Alternative**: Median for robustness

3. **How many outliers?**
   - **Decision**: Top 5% (default)
   - **Rationale**: Standard threshold, balances discovery vs. false positives
   - **Alternative**: Top N for specific use cases

---

## Step 5: Video Generation (`generate_cluster_videos.R`, `find_outliers.R`)

### What It Does

Generates video montages for clusters and outliers using the Python video clipper.

### Why This Step Is Necessary

**Problem**:
- Statistical analysis identifies patterns, but need visual confirmation
- Videos allow behavioral validation
- Communication tool for results

**Solution**:
- Create annotation files for each cluster/outlier group
- Call Python video clipper to generate videos
- Use same standards for consistency

### Process

1. **Group bouts** by cluster ID or outlier status
2. **Create annotation files** in expected format
3. **Call video clipper** with appropriate parameters
4. **Generate videos** with consistent standards

### Video Standards

All videos use the same standards from `generate_bouts_video.py`:
- **Codec**: libx264 + aac
- **Frame rate**: 30 fps
- **Preset**: fast
- **Font**: Helvetica, size 20
- **Bounding boxes**: Yellow outline
- **Text overlay**: Bottom center

**Why consistent standards?**
- **Comparability**: Can compare videos directly
- **Quality**: Professional appearance
- **Reproducibility**: Same settings across all videos

### Output

- Cluster videos: `cluster_{method}_{id}.mp4`
- Outlier video: `outliers.mp4`
- Video name mapping files for reference

### Key Decisions

1. **One video per cluster?**
   - **Decision**: Yes
   - **Rationale**: Allows comparison of cluster behaviors
   - **Alternative**: Single video with all clusters (harder to compare)

2. **Video standards?**
   - **Decision**: Match `generate_bouts_video.py`
   - **Rationale**: Consistency, quality, reproducibility

---

## Integration and Workflow

### Typical Workflow

```bash
# 1. Extract features
Rscript analysis_r/extract_bout_features.R \
  --behavior turn_left \
  --output bout_features.csv

# 2. Cluster bouts
Rscript analysis_r/cluster_bouts.R \
  --input bout_features.csv \
  --method all

# 3. Visualize clusters
Rscript analysis_r/visualize_clusters.R \
  --features bout_features.csv \
  --clusters cluster_assignments_kmeans.csv

# 4. Generate cluster videos
Rscript analysis_r/generate_cluster_videos.R \
  --clusters cluster_assignments_kmeans.csv \
  --method kmeans

# 5. Find outliers
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv

# 6. Select specific bouts (optional)
Rscript analysis_r/select_bouts.R \
  --clusters cluster_assignments_kmeans.csv \
  --cluster-ids 0,1
```

### Data Flow

```
HDF5 Feature Files
    ↓
extract_bout_features.R
    ↓
bout_features.csv (feature matrix)
    ↓
    ├─→ cluster_bouts.R → cluster_assignments.csv
    │                        ↓
    │                   generate_cluster_videos.R → cluster videos
    │
    ├─→ visualize_clusters.R → visualization plots
    │
    └─→ find_outliers.R → outliers.csv + outlier video + plots
```

### Dependencies

- **Feature extraction** must run first (creates input for all other steps)
- **Clustering** needed for cluster videos and visualization
- **Outlier detection** is independent (can run anytime after feature extraction)
- **Video generation** requires clustering or outlier detection results

---

## Statistical Best Practices

### 1. Reproducibility

- Set random seeds for stochastic algorithms
- Document all parameters
- Version control code and data

### 2. Validation

- Use multiple algorithms (K-means, hierarchical, DBSCAN)
- Use internal validation metrics (silhouette, CH index)
- Visual inspection of results
- Domain knowledge validation

### 3. Interpretation

- Multiple perspectives (different algorithms, visualizations)
- Connect statistical findings to behavior
- Acknowledge limitations and assumptions

### 4. Reporting

- Report all methods and parameters
- Include key visualizations
- Report quality metrics
- Explain statistical rationale

---

## Summary

This pipeline provides a comprehensive analysis workflow:

1. **Feature Extraction**: Aggregates temporal sequences into comparable features
2. **Clustering**: Discovers behavioral subtypes using multiple algorithms
3. **Visualization**: Validates and interprets clustering results
4. **Outlier Detection**: Identifies unusual behaviors
5. **Video Generation**: Creates visual representations for validation

Each step is statistically justified and follows best practices for unsupervised learning and exploratory data analysis.

