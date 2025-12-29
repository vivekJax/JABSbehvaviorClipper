# Outlier Detection Explanation

## Why These Bouts Are Considered Outliers

The outlier detection algorithm identifies behavior bouts that are **unlike the majority** of other bouts based on their feature profiles. Here's how it works and what the visualizations show:

## How Outlier Detection Works

### 1. Feature Preprocessing (Optional: PCA Reduction)

**Problem**: Many behavioral features are correlated, and high-dimensional spaces can make distance metrics less meaningful.

**Solution**: Optionally apply PCA reduction before distance calculation:
- **Removes correlation**: Principal components are uncorrelated by construction
- **Reduces dimensionality**: Focuses on directions of maximum variance
- **Improves distance metrics**: Works better in lower-dimensional, uncorrelated space
- **Noise reduction**: Lower components often capture noise

**When to use**: 
- Features are correlated (common in behavioral data)
- High-dimensional data (many features relative to samples)
- Want computational efficiency

**Example**: With 183 features, PCA can reduce to ~43 components while retaining 95% variance.

### 2. Distance Calculation

We compute pairwise distances between all bouts using:
- **Euclidean distance** (default): Standard geometric distance in feature space
  - Assumes features are uncorrelated
  - Works well after scaling
- **Manhattan distance**: Sum of absolute differences
  - More robust to outliers
- **Cosine distance**: Based on angle between feature vectors
  - Scale-invariant, direction-based
- **Mahalanobis distance**: Accounts for covariance structure
  - **Statistically principled**: Incorporates feature correlations
  - **More accurate**: When features are correlated, gives more accurate distances
  - **Formula**: d(x,y) = √((x-y)^T * S^(-1) * (x-y)) where S is covariance matrix

**Why Mahalanobis?**
- Behavioral features are often correlated (e.g., speed and velocity components)
- Euclidean distance treats correlated features as independent, inflating their contribution
- Mahalanobis distance accounts for this correlation structure
- More appropriate for multivariate data with correlations

### 3. Aggregate Distance Metric

For each bout, we calculate an **aggregate distance** that measures how different it is from all other bouts:

- **Mean Distance** (default): Average distance to all other bouts
- **Median Distance**: Median distance to all other bouts  
- **Max Distance**: Maximum distance to any other bout
- **KNN Distance**: Average distance to k nearest neighbors

Bouts with **high aggregate distances** are considered outliers because they are far from the typical behavior pattern.

### 3. Outlier Selection

Outliers are selected based on:
- **Top 5%** (default): Bouts with highest 5% of aggregate distances
- **Top N**: Select the N most distant bouts
- **Percentile threshold**: Custom percentile (e.g., top 10% = 0.9)

## Visualization Explanations

### 1. Distance Distribution Histogram (`outliers_analysis.png`)

**What it shows:**
- Distribution of aggregate distances across all bouts
- Red bars = outliers (high distances)
- Gray bars = normal bouts (typical distances)
- Red dashed line = outlier threshold

**Why it matters:**
Outliers appear in the **right tail** of the distribution, showing they have unusually high distances compared to the majority of bouts.

### 2. Distance Ranking Plot (`outliers_analysis.png`)

**What it shows:**
- All bouts sorted by aggregate distance (lowest to highest)
- Red points = outliers (at the top of the ranking)
- Gray points = normal bouts

**Why it matters:**
This clearly shows outliers **stand out** at the top of the distance ranking, making it obvious why they're considered unusual.

### 3. Box Plot Comparison (`outliers_analysis.png`)

**What it shows:**
- Side-by-side comparison of distance distributions
- Left box = normal bouts (lower distances)
- Right box = outliers (higher distances)

**Why it matters:**
The clear separation between the two groups demonstrates that outliers have **significantly higher** aggregate distances than normal bouts.

### 4. PCA Visualization (`outliers_pca.png`)

**What it shows:**
- 2D projection of all bouts in feature space using Principal Component Analysis
- Red points = outliers (often separated from main cluster)
- Gray points = normal bouts (form the main cluster)
- Point size = aggregate distance (larger = more distant)

**Why it matters:**
Outliers often appear **separated from the main cluster** in feature space, showing they have different feature profiles. This is a visual confirmation that they're truly different.

### 5. t-SNE Visualization (`outliers_tsne.png`)

**What it shows:**
- Non-linear 2D projection using t-SNE (preserves local structure)
- Red points = outliers
- Gray points = normal bouts
- Point size = aggregate distance

**Why it matters:**
t-SNE can reveal **non-linear patterns** that PCA might miss. Outliers that appear separated here are genuinely different in the high-dimensional feature space.

### 6. Feature Comparison (`outliers_features.png`)

**What it shows:**
- Top 10 features with largest differences between outliers and normal bouts
- Red bars = mean feature values for outliers
- Gray bars = mean feature values for normal bouts

**Why it matters:**
This shows **which specific features** make outliers different. For example, outliers might have:
- Different movement speeds
- Different body angles
- Different interaction patterns
- Different durations

### 7. Summary Statistics (`outliers_summary.png`)

**What it shows:**
- Mean aggregate distance for normal vs outlier bouts
- Multiplier showing how much higher outlier distances are

**Why it matters:**
Provides a **quantitative summary** of the difference. For example, if outliers have 2x higher mean distance, they're clearly distinct from the norm.

## Interpreting the Results

### What Makes a Bout an Outlier?

Outliers are bouts that have:
1. **Unusual feature combinations** - Different from typical behavior patterns
2. **High distance to others** - Far from the center of the data distribution
3. **Distinct characteristics** - Visible separation in feature space visualizations

### Common Reasons for Outliers

- **Atypical movement patterns** - Different speed, direction, or trajectory
- **Unusual durations** - Much shorter or longer than typical bouts
- **Rare behavioral contexts** - Occurring in unusual situations
- **Measurement artifacts** - Edge cases in data collection
- **Genuinely rare behaviors** - True behavioral anomalies worth investigating

### Using Outlier Information

1. **Review the video** (`outliers.mp4`) to see what makes them visually different
2. **Check feature values** (`outliers.csv`) to see which features are unusual
3. **Compare with clusters** - Are outliers similar to any cluster, or truly unique?
4. **Investigate context** - What conditions led to these unusual behaviors?

## Example Interpretation

If you see:
- **High aggregate distance** (e.g., 26.1 vs 15.2 for normal bouts)
- **Separation in PCA/t-SNE** (red points away from gray cluster)
- **Different feature values** (e.g., much higher speed, different angles)

Then these bouts are **genuinely different** from typical behavior and worth investigating further.

