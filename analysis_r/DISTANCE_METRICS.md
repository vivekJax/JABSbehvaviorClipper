# Distance Metrics for Outlier Detection

## Overview

This document explains the different distance metrics available for outlier detection and when to use each one. The choice of distance metric is crucial because it determines how "different" is measured in feature space.

## The Problem: Feature Correlations

Behavioral features are often **correlated**. For example:
- Speed and velocity magnitude are highly correlated
- Body angles may be correlated with movement direction
- Aggregated statistics (mean, std) from the same underlying feature are correlated

**Why this matters**:
- **Euclidean distance** treats all features as independent
- If two features are correlated, Euclidean distance **double-counts** their effect
- This can make distances misleading and outlier detection less accurate

## Distance Metrics

### 1. Euclidean Distance (Default)

```
d(x, y) = √Σ(xi - yi)²
```

**Characteristics**:
- Standard geometric distance
- Assumes features are **uncorrelated** and **isotropic** (equal importance)
- Fast computation: O(n²p) where n=samples, p=features

**When to use**:
- Features are approximately uncorrelated
- After PCA reduction (components are uncorrelated)
- For computational efficiency
- As a baseline for comparison

**Limitations**:
- **Ignores correlations**: Treats correlated features as independent
- **Example**: If speed and velocity are correlated, Euclidean distance inflates their contribution

### 2. Manhattan Distance

```
d(x, y) = Σ|xi - yi|
```

**Characteristics**:
- L1 norm (vs L2 for Euclidean)
- More robust to outliers
- Less sensitive to extreme values

**When to use**:
- Want robustness to outliers
- Sparse data
- As alternative to Euclidean

**Limitations**:
- Still ignores correlations
- Less intuitive than Euclidean

### 3. Cosine Distance

```
d(x, y) = 1 - (x·y) / (||x|| ||y||)
```

**Characteristics**:
- Based on angle between vectors, not magnitude
- Scale-invariant
- Focuses on direction, not distance

**When to use**:
- Want scale-invariant distances
- Interested in pattern similarity, not magnitude
- High-dimensional data (often works better)

**Limitations**:
- Doesn't account for magnitude differences
- Still ignores correlations

### 4. Mahalanobis Distance ⭐ **Recommended for Correlated Features**

```
d(x, y) = √((x - y)^T * S^(-1) * (x - y))
```

where S is the covariance matrix of the features.

**Characteristics**:
- **Accounts for correlations**: Incorporates covariance structure
- **Statistically principled**: Based on multivariate normal distribution
- **Scale-invariant**: Already accounts for different scales
- **More accurate**: When features are correlated, gives more accurate distances

**When to use**:
- ✅ **Features are correlated** (common in behavioral data)
- ✅ Want statistically principled approach
- ✅ Want to account for covariance structure
- ✅ More accurate outlier detection

**Computational considerations**:
- Requires computing covariance matrix: O(np²)
- Requires matrix inversion: O(p³)
- For high-dimensional data, may need PCA reduction first
- Uses pseudo-inverse if covariance matrix is near-singular

**Example**:
- If speed and velocity magnitude are highly correlated (r=0.9)
- Euclidean distance treats them as independent: d² = (speed_diff)² + (vel_diff)²
- But they're not independent! Mahalanobis accounts for this: d² = (diff)^T * S^(-1) * (diff)
- Result: More accurate distance that doesn't double-count correlated information

## PCA-Based Distance Calculation

### The Approach

Instead of using Mahalanobis distance directly, we can:
1. Apply PCA to remove correlations
2. Calculate distances in PCA space (where components are uncorrelated)
3. Euclidean distance in PCA space is appropriate (components are uncorrelated)

### Why This Works

**PCA properties**:
- Principal components are **uncorrelated** by construction
- Components are **orthogonal** (independent directions)
- Retains maximum variance in fewer dimensions

**Result**:
- After PCA, features (components) are uncorrelated
- Euclidean distance in PCA space is appropriate
- Equivalent to Mahalanobis in original space (if using all components)

### Advantages

1. **Removes correlation**: Components are uncorrelated
2. **Reduces dimensionality**: Focuses on signal, reduces noise
3. **Computational efficiency**: Fewer dimensions = faster computation
4. **Curse of dimensionality**: Distance metrics work better in lower dimensions
5. **Noise reduction**: Lower components often capture noise

### When to Use

- ✅ **High-dimensional data**: Many features relative to samples
- ✅ **Correlated features**: Want to remove correlation
- ✅ **Computational constraints**: Need faster computation
- ✅ **Noise reduction**: Want to focus on signal, not noise

### Variance Threshold

**Default: 0.95** (retain 95% variance)

**Rationale**:
- Retains most information while reducing dimensionality
- Typically reduces dimensions significantly (e.g., 183 → 43 features)
- Lower components often capture noise

**Trade-off**:
- Higher threshold = more dimensions = more information but less reduction
- Lower threshold = fewer dimensions = more reduction but potential information loss

## Comparison: Mahalanobis vs PCA + Euclidean

### Mahalanobis Distance

**Pros**:
- Accounts for full covariance structure
- Statistically principled
- No information loss (uses all features)

**Cons**:
- Computationally expensive for high dimensions
- Requires invertible covariance matrix
- Doesn't reduce dimensionality

### PCA + Euclidean

**Pros**:
- Removes correlation (components are uncorrelated)
- Reduces dimensionality (faster computation)
- Noise reduction
- Works well in high dimensions

**Cons**:
- Information loss (if variance threshold < 1.0)
- Need to choose variance threshold
- Less direct than Mahalanobis

### Recommendation

**For correlated features**:
- **Use Mahalanobis** if: p < 100 features, want full covariance structure
- **Use PCA + Euclidean** if: p > 100 features, want dimensionality reduction, computational efficiency

**For high-dimensional data**:
- **Use PCA + Euclidean**: Addresses both correlation and curse of dimensionality

**Note**: Using both (PCA + Mahalanobis) is redundant because PCA components are uncorrelated, so Mahalanobis ≈ Euclidean in PCA space.

## Statistical Justification

### Why Mahalanobis?

**Multivariate Normal Distribution**:
- If features follow multivariate normal: X ~ N(μ, Σ)
- Mahalanobis distance: D² = (X - μ)^T * Σ^(-1) * (X - μ)
- This follows a chi-square distribution: D² ~ χ²(p)
- Statistically principled outlier detection

**Geometric Interpretation**:
- Mahalanobis distance measures distance in **standardized, decorrelated space**
- Accounts for different scales and correlations
- More accurate than Euclidean for correlated data

### Why PCA?

**Principal Component Analysis**:
- Finds directions of maximum variance
- Components are uncorrelated by construction
- Optimal linear transformation for variance preservation

**Dimensionality Reduction**:
- Retains most information in fewer dimensions
- Addresses curse of dimensionality
- Improves distance metric performance

## Usage Examples

### Example 1: Mahalanobis Distance

```bash
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --distance-metric mahalanobis \
  --output-dir outlier_videos_mahalanobis
```

**Use when**: Features are correlated, want to account for covariance structure

### Example 2: PCA Reduction + Euclidean

```bash
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --use-pca \
  --pca-variance 0.95 \
  --output-dir outlier_videos_pca
```

**Use when**: High-dimensional data, want dimensionality reduction, correlated features

### Example 3: Standard Euclidean (Baseline)

```bash
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --distance-metric euclidean \
  --output-dir outlier_videos_euclidean
```

**Use when**: Features are approximately uncorrelated, baseline comparison

## Summary

| Method | Accounts for Correlation | Dimensionality Reduction | Computational Cost | Best For |
|--------|------------------------|-------------------------|-------------------|----------|
| Euclidean | ❌ | ❌ | Low | Uncorrelated features |
| Manhattan | ❌ | ❌ | Low | Robust to outliers |
| Cosine | ❌ | ❌ | Low | Scale-invariant, high-dim |
| Mahalanobis | ✅ | ❌ | High | Correlated features, p < 100 |
| PCA + Euclidean | ✅ | ✅ | Medium | High-dim, correlated features |

**Recommendation for behavioral data**:
- **Default**: Use **Mahalanobis** or **PCA + Euclidean** (features are typically correlated)
- **High-dimensional**: Use **PCA + Euclidean** (addresses both correlation and dimensionality)
- **Baseline**: Use **Euclidean** for comparison

