---
title: "Technical Guide - Statistical Methodology"
description: "Comprehensive statistical and methodological explanations"
category: "technical"
audience: ["analysts", "researchers", "developers"]
tags: ["statistics", "methodology", "clustering", "outliers", "distance-metrics"]
documentation_id: "r-technical-guide"
---

# Technical Guide: Statistical Methodology

## Overview

This document provides comprehensive statistical and methodological explanations for the behavior bout analysis pipeline. Each method is explained with statistical rationale, assumptions, and best practices.

## Table of Contents

1. [Feature Extraction](#feature-extraction)
2. [Data Preprocessing](#data-preprocessing)
3. [Clustering Analysis](#clustering-analysis)
4. [Outlier Detection](#outlier-detection)
5. [Visualization Strategy](#visualization-strategy)
6. [Statistical Considerations](#statistical-considerations)

---

## Feature Extraction

### Purpose

Behavior bouts are characterized by high-dimensional per-frame features (angles, velocities, distances, etc.). To enable statistical analysis, we need to **aggregate** these temporal sequences into bout-level summary statistics.

### Statistical Rationale

**Why aggregate?**
- Per-frame features are **temporal sequences** (time series), not single values
- Most statistical methods (clustering, distance metrics) require **fixed-length vectors**
- Aggregation reduces dimensionality while preserving important information
- Enables comparison across bouts of different durations

### Aggregation Strategy

We compute multiple statistics for each feature:

1. **Mean**: Central tendency - captures typical behavior
2. **Standard Deviation**: Variability - captures consistency vs. variability
3. **Min/Max**: Range - captures extreme values
4. **Median**: Robust central tendency - less sensitive to outliers
5. **First/Last**: Temporal dynamics - captures behavior change over time
6. **Duration**: Temporal extent - bout length as a feature

**Why these statistics?**
- **Mean**: Standard measure of central tendency, captures overall behavior level
- **Std**: Important for distinguishing consistent vs. variable behaviors
- **Min/Max**: Captures extreme behaviors that might be behaviorally significant
- **Median**: Robust alternative to mean, less affected by outliers
- **First/Last**: Captures temporal dynamics (e.g., acceleration, deceleration)
- **Duration**: Behaviorally meaningful - different bout lengths may indicate different behaviors

### Statistical Considerations

- **Information Loss**: Aggregation loses temporal ordering information
  - *Trade-off*: Necessary for statistical analysis, but temporal patterns are preserved in first/last values
- **Feature Selection**: All available features are included
  - *Rationale*: Let the data-driven methods (clustering, PCA) identify important features
- **Missing Data**: Handled during preprocessing (see below)

---

## Data Preprocessing

### Step 1: Missing Value Handling

**Problem**: HDF5 feature files may have missing values (NaN) due to:
- Tracking failures
- Edge cases in feature computation
- Data collection artifacts

**Solution**: Mean imputation

**Why mean imputation?**
- **Preserves sample size**: Dropping rows would reduce statistical power
- **Unbiased for MCAR**: If missing is "Missing Completely At Random", mean imputation is unbiased
- **Conservative**: Mean imputation doesn't introduce extreme values
- **Standard practice**: Common in high-dimensional data analysis

**Alternative considered**: Median imputation
- More robust to outliers, but we use mean for consistency with clustering assumptions

### Step 2: Feature Scaling

**Problem**: Features have different scales (e.g., angles in degrees, distances in pixels, velocities in pixels/frame)

**Why scale?**
- **Distance metrics are scale-dependent**: Euclidean distance is dominated by features with larger scales
- **Clustering algorithms are scale-sensitive**: K-means, hierarchical clustering assume equal feature importance
- **Prevents bias**: Without scaling, some features dominate the analysis

**Scaling Methods**:

1. **Standard Scaling (Z-score normalization)** - Default
   ```
   z = (x - μ) / σ
   ```
   - **Mean**: 0, **Std**: 1
   - **Rationale**: Most common, works well for normally distributed features
   - **Assumption**: Features are approximately normally distributed
   - **Use case**: General-purpose, works for most scenarios

2. **Min-Max Scaling**
   ```
   z = (x - min) / (max - min)
   ```
   - **Range**: [0, 1]
   - **Rationale**: Preserves relative distances, bounded range
   - **Use case**: When you need bounded features (e.g., neural networks)

3. **Robust Scaling**
   ```
   z = (x - median) / MAD
   ```
   - **Rationale**: Uses median and MAD (Median Absolute Deviation), robust to outliers
   - **Use case**: When data has many outliers

**Why standard scaling by default?**
- Most widely used and understood
- Works well with distance-based methods
- Assumes normal distribution (reasonable for aggregated statistics)

### Step 3: Constant Feature Removal

**Problem**: Some features may have zero or near-zero variance (constant across all bouts)

**Why remove?**
- **No information**: Constant features don't help distinguish bouts
- **Numerical issues**: Can cause problems in PCA, matrix inversions
- **Computational efficiency**: Reduces dimensionality

**Threshold**: Variance > 1e-10
- **Rationale**: Accounts for floating-point precision while removing truly constant features

### Step 4: Infinite/NaN Handling

**Problem**: Scaling or feature computation may produce infinite or NaN values

**Solution**: Replace with 0

**Why 0?**
- **Conservative**: Doesn't introduce extreme values
- **Neutral**: 0 is the mean after standard scaling
- **Prevents errors**: Many algorithms fail on infinite/NaN values

---

## Clustering Analysis

### Purpose

Identify **groups** of similar behavior bouts. This helps:
- Discover behavioral subtypes
- Understand behavioral diversity
- Guide further analysis

### Statistical Framework

Clustering is an **unsupervised learning** problem:
- **No ground truth**: We don't know the "correct" clusters
- **Exploratory**: Goal is to discover patterns
- **Validation**: Use internal metrics (silhouette, Calinski-Harabasz)

### Clustering Algorithms

#### 1. K-Means Clustering

**Method**:
- Partition data into k clusters
- Minimize within-cluster sum of squares (WCSS)
- Iterative optimization (Lloyd's algorithm)

**Why K-means?**
- **Fast**: O(nkd) complexity, efficient for large datasets
- **Interpretable**: Each cluster has a clear centroid
- **Scalable**: Works well with many samples
- **Standard**: Most widely used clustering algorithm

**Limitations**:
- Assumes **spherical clusters** (equal variance in all directions)
- Requires **k** to be specified
- Sensitive to **initialization** (solved by multiple restarts)

**Optimal k Selection**:
- **Elbow method**: Plot WCSS vs k, find "elbow"
- **Silhouette score**: Measure of cluster quality
  - Range: [-1, 1]
  - Higher = better separation
  - We use this for automatic k selection

**Why silhouette score?**
- **Internal validation**: Doesn't require ground truth
- **Balanced**: Considers both cohesion (within-cluster) and separation (between-cluster)
- **Standard metric**: Widely accepted in clustering literature

#### 2. Hierarchical Clustering (Agglomerative)

**Method**:
- Start with each point as its own cluster
- Iteratively merge closest clusters
- Creates a dendrogram (tree structure)

**Linkage Methods**:
- **Ward's method** (default): Minimizes within-cluster variance
  - **Why Ward?**: Produces compact, spherical clusters (similar to K-means)
  - **Best for**: Well-separated clusters of similar size

**Why hierarchical?**
- **No k required**: Can examine dendrogram to choose k
- **Flexible**: Can handle non-spherical clusters
- **Interpretable**: Dendrogram shows cluster relationships
- **Complementary**: Different assumptions than K-means

**Limitations**:
- **Computational cost**: O(n² log n) or O(n³) depending on linkage
- **Greedy**: Merges are final (no backtracking)

#### 3. DBSCAN

**Method**:
- Density-based clustering
- Groups points in dense regions
- Marks sparse points as noise

**Parameters**:
- **eps**: Maximum distance for neighbors
- **minPts**: Minimum points to form a cluster

**Why DBSCAN?**
- **No k required**: Discovers number of clusters automatically
- **Handles noise**: Identifies outliers as noise points
- **Flexible shapes**: Can find non-spherical clusters
- **Robust**: Less sensitive to initialization

**Limitations**:
- **Parameter sensitive**: eps and minPts are critical
- **Density variation**: Struggles with varying densities
- **High-dimensional**: "Curse of dimensionality" affects distance metrics

**Why we include it?**
- Provides alternative perspective
- Can identify noise/outliers
- Useful when clusters have irregular shapes

### Cluster Validation

**Silhouette Score**:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
- **a(i)**: Average distance to points in same cluster
- **b(i)**: Average distance to points in nearest other cluster
- **Interpretation**: 
  - s(i) ≈ 1: Well-clustered
  - s(i) ≈ 0: On cluster boundary
  - s(i) ≈ -1: Mis-clustered

**Calinski-Harabasz Index** (Variance Ratio):
```
CH = (SSB / (k-1)) / (SSW / (n-k))
```
- **SSB**: Between-cluster sum of squares
- **SSW**: Within-cluster sum of squares
- **Higher = better**: More separation, less within-cluster variance

**Why both metrics?**
- **Silhouette**: Focuses on individual point placement
- **CH Index**: Focuses on overall cluster structure
- **Complementary**: Different aspects of cluster quality

---

## Outlier Detection

### Purpose

Identify behavior bouts that are **unlike the majority**. These may represent:
- Rare behaviors
- Measurement errors
- Atypical patterns worth investigating

### Statistical Framework

Outlier detection uses **distance-based methods**:
- Compute distances between all pairs of bouts
- Identify bouts that are far from others
- Use aggregate distance metrics to rank outliers

### PCA-Based Distance Calculation

**Problem**: High-dimensional feature spaces with correlated features

**Why use PCA before distance calculation?**
1. **Removes correlation**: Principal components are uncorrelated by construction
2. **Reduces dimensionality**: Focuses on directions of maximum variance
3. **Curse of dimensionality**: Distance metrics become less meaningful in high dimensions
4. **Computational efficiency**: Fewer dimensions = faster computation
5. **Noise reduction**: Lower components often capture noise

**How it works**:
1. Perform PCA on scaled features
2. Select components that retain specified variance (default: 95%)
3. Calculate distances in reduced PCA space
4. Distances are now in uncorrelated, variance-maximizing space

**When to use**:
- **High-dimensional data**: Many features relative to samples
- **Correlated features**: Features are not independent
- **Computational constraints**: Need faster computation
- **Noise reduction**: Want to focus on signal, not noise

**Variance threshold**:
- **Default: 0.95** (95% variance retained)
- **Rationale**: Retains most information while reducing dimensionality
- **Trade-off**: Higher threshold = more dimensions = more information but less reduction

**Combined with Mahalanobis**:
- PCA removes correlation (components are uncorrelated)
- Mahalanobis in PCA space is equivalent to Euclidean (since components are uncorrelated)
- But PCA reduction itself addresses the correlation issue
- **Recommendation**: Use PCA reduction OR Mahalanobis, not both (redundant)

### Distance Metrics

#### 1. Euclidean Distance (Default)

```
d(x, y) = √Σ(xi - yi)²
```

**Why Euclidean?**
- **Standard**: Most widely used distance metric
- **Intuitive**: Geometric distance in feature space
- **Works well**: After scaling, assumes features are equally important
- **Fast**: Computationally efficient

**Assumptions**:
- Features are **isotropic** (equal importance in all directions)
- Features are **uncorrelated** (orthogonal)
- After scaling, this is reasonable if features are approximately uncorrelated

**Limitations**:
- **Ignores correlations**: If features are correlated, Euclidean distance can be misleading
- **Example**: If two features are highly correlated, Euclidean distance double-counts their effect

#### 4. Mahalanobis Distance

```
d(x, y) = √((x - y)^T * S^(-1) * (x - y))
```

where S is the covariance matrix of the features.

**Why Mahalanobis?**
- **Accounts for correlations**: Incorporates covariance structure
- **Scale-invariant**: Already accounts for different scales
- **Statistically principled**: Based on multivariate normal distribution
- **Better for correlated features**: More appropriate when features are correlated

**When to use**:
- Features are **correlated** (common in behavioral data)
- You want to account for **covariance structure**
- You want **statistically principled** distance metric

**Example**: 
- If two features are highly correlated (e.g., speed and velocity magnitude)
- Euclidean distance treats them as independent, inflating their contribution
- Mahalanobis distance accounts for this correlation, giving more accurate distances

**Computational considerations**:
- Requires computing and inverting covariance matrix: O(p³) where p = number of features
- For high-dimensional data, may need regularization or PCA reduction first
- Uses pseudo-inverse if covariance matrix is near-singular

#### 2. Manhattan Distance

```
d(x, y) = Σ|xi - yi|
```

**Why include?**
- **Robust**: Less sensitive to outliers (L1 vs L2 norm)
- **Sparse data**: Can work better with sparse features
- **Alternative perspective**: Different from Euclidean

#### 3. Cosine Distance

```
d(x, y) = 1 - (x·y) / (||x|| ||y||)
```

**Why include?**
- **Scale-invariant**: Only considers direction, not magnitude
- **High-dimensional**: Often works better in high dimensions
- **Complementary**: Captures different aspects of similarity

### Aggregate Distance Metrics

For each bout, we compute an aggregate distance to all other bouts:

#### 1. Mean Distance (Default)

```
D_mean(i) = (1/(n-1)) Σ d(i, j)
```

**Why mean?**
- **Comprehensive**: Considers all other bouts
- **Stable**: Less sensitive to individual outliers
- **Interpretable**: Average distance is intuitive
- **Standard**: Most common aggregate metric

#### 2. Median Distance

```
D_median(i) = median({d(i, j) : j ≠ i})
```

**Why include?**
- **Robust**: Less sensitive to extreme distances
- **Alternative**: When mean is affected by outliers

#### 3. Maximum Distance

```
D_max(i) = max({d(i, j) : j ≠ i})
```

**Why include?**
- **Extreme**: Identifies bouts far from their nearest neighbor
- **Conservative**: Ensures outlier is far from at least one other bout

#### 4. K-Nearest Neighbors Distance

```
D_knn(i) = (1/k) Σ d(i, j) for j in k nearest neighbors
```

**Why include?**
- **Local**: Focuses on local neighborhood
- **Robust**: Less affected by distant points
- **Flexible**: k = √n adapts to sample size

### Outlier Selection

**Top 5% (Default)**:
- Select bouts with highest 5% of aggregate distances
- **Rationale**: 
  - Standard threshold in outlier detection
  - Balances sensitivity (finding outliers) vs. specificity (avoiding false positives)
  - Adapts to data distribution

**Top N**:
- Select exactly N most distant bouts
- **Use case**: When you need a specific number

**Percentile Threshold**:
- Custom percentile (e.g., top 10% = 0.9)
- **Use case**: When you have domain knowledge about expected outlier rate

**Why percentile-based?**
- **Distribution-agnostic**: Works regardless of distance distribution
- **Adaptive**: Adjusts to data characteristics
- **Standard**: Common in statistical practice

### Statistical Considerations

**Multiple Testing**:
- We're identifying multiple outliers simultaneously
- **Issue**: Risk of false positives increases with number of tests
- **Mitigation**: Using aggregate distances (single metric per bout) reduces multiple testing issues

**Distance Matrix Computation**:
- **Complexity**: O(n²) for n bouts
- **Scalability**: For very large datasets, consider sampling or approximate methods
- **Memory**: Full distance matrix requires O(n²) memory

**Feature Scaling Importance**:
- **Critical**: Without scaling, distance metrics are dominated by high-variance features
- **Solution**: Standard scaling ensures equal feature importance

**Feature Correlation Considerations**:
- **Problem**: Many behavioral features are correlated (e.g., speed and velocity components)
- **Impact**: Euclidean distance treats correlated features as independent, inflating their contribution
- **Solutions**:
  1. **Mahalanobis distance**: Accounts for covariance structure directly
  2. **PCA reduction**: Removes correlation by using uncorrelated components
  3. **Combined approach**: PCA reduction + Euclidean (components are uncorrelated)

**Recommendation**:
- **For correlated features**: Use Mahalanobis distance OR PCA reduction
- **For high-dimensional data**: Use PCA reduction to reduce dimensionality
- **For computational efficiency**: Use PCA reduction (fewer dimensions)
- **For statistical rigor**: Use Mahalanobis distance (accounts for full covariance structure)

---

## Visualization Strategy

### Purpose

Visualizations serve multiple purposes:
1. **Exploratory**: Discover patterns in data
2. **Validation**: Verify clustering/outlier detection results
3. **Communication**: Explain results to others
4. **Quality Control**: Identify issues in analysis

### Dimensionality Reduction

#### Principal Component Analysis (PCA)

**Method**:
- Linear transformation to lower dimensions
- Preserves maximum variance
- Orthogonal components

**Why PCA?**
- **Linear**: Simple, interpretable transformation
- **Variance-preserving**: Captures most information in first few components
- **Standard**: Most widely used dimensionality reduction
- **Fast**: Computationally efficient

**Interpretation**:
- **PC1, PC2**: First two principal components capture most variance
- **Variance explained**: Percentage shows how much information is retained
- **Limitation**: Assumes linear relationships

**When to use**:
- Initial exploration
- When relationships are approximately linear
- When interpretability is important

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Method**:
- Non-linear dimensionality reduction
- Preserves local structure (neighborhoods)
- Stochastic optimization

**Why t-SNE?**
- **Non-linear**: Can capture complex relationships
- **Local structure**: Preserves neighborhoods well
- **Visualization**: Excellent for 2D visualization
- **Complementary**: Different from PCA

**Limitations**:
- **Non-interpretable**: Components don't have clear meaning
- **Parameter sensitive**: Perplexity parameter affects results
- **Computational cost**: Slower than PCA
- **Global structure**: May not preserve global distances

**When to use**:
- When PCA shows poor separation
- When you suspect non-linear relationships
- For visualization purposes

**Perplexity Selection**:
- **Default**: 30
- **Adaptive**: We use min(30, (n-1)/3) for small samples
- **Rationale**: Perplexity must be < n, and should scale with sample size

### Visualization Types

#### 1. Distance Distribution Histogram

**Purpose**: Show distribution of aggregate distances

**Why useful?**
- **Identify threshold**: Visual confirmation of outlier threshold
- **Distribution shape**: Understand distance distribution
- **Outlier visibility**: Outliers appear in right tail

**Statistical interpretation**:
- **Right-skewed**: Common in distance distributions
- **Threshold line**: Shows where outliers are selected
- **Separation**: Clear separation indicates good outlier detection

#### 2. Distance Ranking Plot

**Purpose**: Show all bouts ranked by distance

**Why useful?**
- **Ranking**: Clear visualization of distance ordering
- **Outlier identification**: Outliers at top of ranking
- **Gap analysis**: Large gaps indicate natural breakpoints

#### 3. Box Plot Comparison

**Purpose**: Compare distance distributions between groups

**Why useful?**
- **Statistical comparison**: Visual test of group differences
- **Distribution shape**: Shows median, quartiles, outliers
- **Effect size**: Visual representation of difference magnitude

**Statistical interpretation**:
- **Non-overlapping boxes**: Strong evidence of difference
- **Median difference**: Robust measure of central tendency difference
- **Whiskers**: Show range of data

#### 4. Feature Space Plots (PCA/t-SNE)

**Purpose**: Show spatial relationships in feature space

**Why useful?**
- **Separation**: Visual confirmation of cluster/outlier separation
- **Structure**: Reveals data structure (clusters, manifolds)
- **Validation**: Confirms results from clustering algorithms

**Interpretation**:
- **Clusters**: Groups of points close together
- **Outliers**: Points far from main cluster
- **Overlap**: Indicates similar features (may be mis-clustered)

#### 5. Feature Comparison Plots

**Purpose**: Show which features distinguish groups

**Why useful?**
- **Interpretability**: Understand what makes groups different
- **Feature importance**: Identify most discriminative features
- **Biological relevance**: Connect statistical differences to behavior

**Statistical interpretation**:
- **Large differences**: Features that strongly distinguish groups
- **Small differences**: Features that are similar across groups
- **Direction**: Whether outliers are higher or lower

---

## Statistical Considerations

### Assumptions and Limitations

#### 1. Independence

**Assumption**: Bouts are independent observations

**Reality**: 
- Bouts from same animal may be correlated
- Bouts from same video may share context

**Impact**:
- **Clustering**: May create clusters based on animal/video rather than behavior
- **Outlier detection**: May identify animal-specific patterns as outliers

**Mitigation**:
- Include animal/video as metadata for interpretation
- Consider hierarchical models if needed

#### 2. Feature Selection

**Current approach**: Use all available features

**Rationale**:
- **Data-driven**: Let algorithms identify important features
- **Comprehensive**: Don't lose potentially important information
- **Standard**: Common in high-dimensional analysis

**Alternative**: Feature selection
- **Pros**: Reduces noise, improves interpretability
- **Cons**: Risk of losing important features, adds complexity

#### 3. Sample Size

**Considerations**:
- **Clustering**: Need sufficient samples per cluster (rule of thumb: 10-20 per cluster)
- **Outlier detection**: Need sufficient samples to define "normal"
- **PCA/t-SNE**: Need samples > features for stable results

**Current dataset**: 117 bouts
- **Adequate for**: K-means with k=3-5, outlier detection
- **Marginal for**: Hierarchical clustering with many clusters
- **Good for**: Exploratory analysis

#### 4. Multiple Comparisons

**Issue**: When testing multiple hypotheses (e.g., multiple clusters, multiple outliers)

**Current approach**: 
- Use internal validation metrics (silhouette, CH index)
- Use percentile-based thresholds (not p-values)
- Focus on effect size, not significance

**Rationale**:
- **Exploratory**: Goal is discovery, not hypothesis testing
- **Effect size**: Focus on meaningful differences
- **Visual validation**: Use plots to confirm results

### Best Practices

#### 1. Reproducibility

- **Random seeds**: Set for stochastic algorithms (t-SNE, K-means initialization)
- **Parameter documentation**: Document all choices
- **Version control**: Track code and data versions

#### 2. Validation

- **Internal metrics**: Silhouette, CH index
- **Visual inspection**: Always examine plots
- **Domain knowledge**: Validate against behavioral expectations

#### 3. Interpretation

- **Multiple perspectives**: Use multiple algorithms and visualizations
- **Biological relevance**: Connect statistical findings to behavior
- **Uncertainty**: Acknowledge limitations and assumptions

#### 4. Reporting

- **Transparency**: Report all methods and parameters
- **Visualization**: Include key plots
- **Statistics**: Report cluster quality metrics, outlier statistics

---

## Summary

This analysis pipeline follows statistical best practices:

1. **Feature extraction**: Aggregates temporal sequences into comparable statistics
2. **Preprocessing**: Handles missing data, scales features, removes constants
3. **Clustering**: Uses multiple algorithms with internal validation
4. **Outlier detection**: Distance-based methods with multiple metrics
5. **Visualization**: Multiple perspectives for validation and interpretation

Each step is justified by statistical theory and data science principles, with clear rationale for methodological choices.

