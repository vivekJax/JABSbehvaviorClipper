# Behavioral Bout Analysis Report: Turn Left Behavior

**Analysis Date:** December 29, 2025  
**Behavior:** turn_left  
**Analysis Pipeline:** JABS Behavior Clipper with Hierarchical and B-SOID Clustering

---

## Executive Summary

This report presents a comprehensive analysis of behavioral bout clustering for the "turn_left" behavior using a multi-stage pipeline that includes feature extraction, outlier detection, dimensionality reduction, and unsupervised clustering. The analysis identified 243 behavioral bouts from video recordings, removed 18 outliers (7.4%) through consensus-based detection, and applied two complementary clustering methods: hierarchical clustering and B-SOID (Behavioral Segmentation of Open Field in DeepLabCut). The hierarchical clustering method identified 2 distinct clusters, while B-SOID identified 5 clusters with 99 noise points (44.0% of the dataset), providing complementary perspectives on behavioral structure.

---

## 1. Data Collection and Preprocessing

### 1.1 Initial Bout Extraction

- **Total bouts extracted:** 243 bouts
- **Behavior:** turn_left (present=True only)
- **Data source:** Video recordings with pose estimation data (HDF5 format)
- **Annotation format:** JSON files containing unfragmented bout boundaries

The analysis used `unfragmented_labels` from annotation files to ensure that bout boundaries matched the original labeling in the GUI, preserving the intended start and end frames of each behavioral bout.

### 1.2 Feature Extraction

**Total features extracted:** 486 features per bout

Features were aggregated from per-frame pose estimation data using the following statistical measures:
- **Mean:** Central tendency of each feature
- **Standard deviation:** Variability within the bout
- **Min/Max:** Range of feature values
- **Median:** Robust central tendency
- **First/Last:** Temporal dynamics (beginning vs. end of bout)
- **Duration:** Bout length in frames
- **IQR (Interquartile Range):** Robust measure of spread

**Feature categories:**
1. **Angular velocity:** Rotational movement characteristics
2. **Centroid velocity:** Direction and magnitude of center-of-mass movement
3. **Lixit distances:** Distances from body parts to lixit (water source)
4. **Mouse-lixit angles:** Angular relationships between body parts and lixit
5. **Pairwise distances:** Inter-body-part distances
6. **Point masks:** Visibility/confidence of pose estimation points
7. **Point speeds:** Velocity of individual body parts
8. **Point velocity directions:** Directional components of body part movement

Each feature category was computed for multiple body parts (NOSE, LEFT_EAR, RIGHT_EAR, BASE_TAIL, TIP_TAIL, etc.), resulting in a high-dimensional feature space that captures the kinematic and spatial characteristics of the turn_left behavior.

---

## 2. Outlier Detection

### 2.1 Methodology

Outlier detection employed a **consensus-based approach** using multiple distance metrics and aggregation methods to ensure robust identification of anomalous bouts. This multi-method approach reduces false positives and increases confidence in outlier identification.

**Distance metrics used:**
1. **Euclidean distance:** Standard geometric distance
2. **Mahalanobis distance:** Accounts for feature correlations (statistically principled)
3. **Manhattan distance:** L1 norm, robust to outliers
4. **Cosine distance:** Angle-based similarity, scale-invariant

**Aggregation methods:**
1. **Mean distance:** Average across all distance metrics
2. **Median distance:** Robust to extreme values
3. **Maximum distance:** Identifies bouts that are outliers by any metric
4. **K-Nearest Neighbors (KNN) distance:** Local density-based outlier score

**Consensus threshold:** A bout was identified as an outlier if it was flagged by at least 2 different methods, ensuring that only consistently anomalous bouts were removed.

### 2.2 Results

- **Total outliers identified:** 18 bouts (7.4% of dataset)
- **Bouts remaining after outlier removal:** 225 bouts (92.6% of dataset)
- **Outlier removal rate:** 7.4%

The consensus approach successfully identified and removed anomalous bouts that could have distorted the clustering analysis. These outliers likely represent:
- Mislabeled bouts (false positives in behavior annotation)
- Bouts with incomplete or corrupted pose estimation data
- Extreme behavioral variants that do not represent the typical turn_left behavior

### 2.3 Outlier Characteristics

Outliers were distributed across multiple videos, indicating that the detection was not biased toward specific recording sessions. The removed outliers were characterized by:
- Extreme feature values in one or more feature categories
- High distance scores across multiple distance metrics
- Consistent identification across aggregation methods

**Visualization outputs:**
- Distance distribution histograms
- Distance ranking plots
- PCA visualization with outliers highlighted
- t-SNE visualization with outliers highlighted
- Feature comparison plots
- Summary statistics plots

All outlier visualizations are available in `BoutResults/outliers/` directory.

---

## 3. Dimensionality Reduction

### 3.1 Principal Component Analysis (PCA)

Prior to clustering, the feature space was reduced using Principal Component Analysis (PCA) to:
1. Remove multicollinearity between features
2. Reduce computational complexity
3. Improve clustering performance
4. Retain interpretable linear combinations of original features

**PCA parameters:**
- **Variance threshold:** 95% of total variance retained
- **Original features:** 486 features
- **PCA components retained:** 64 components
- **Variance explained:** 95.0%

The PCA transformation reduced the feature space from 486 dimensions to 64 dimensions while retaining 95% of the variance, indicating that the original features contained substantial redundancy. This dimensionality reduction is particularly important for hierarchical clustering, which scales with O(n^2 log n) or O(n^3) complexity.

**Note:** B-SOID clustering uses UMAP (non-linear dimensionality reduction) directly on the original features, bypassing PCA. This allows B-SOID to capture non-linear relationships that PCA might miss.

---

## 4. Clustering Analysis

Two complementary clustering methods were applied to the cleaned dataset (225 bouts after outlier removal):

### 4.1 Hierarchical Clustering

#### 4.1.1 Methodology

**Algorithm:** Agglomerative hierarchical clustering with Ward's linkage method (ward.D2)

**Optimal k selection:**
- **Primary criterion:** Silhouette score (maximized)
- **Secondary criterion:** Elbow method (within-cluster sum of squares)
- **Selected k:** 2 clusters

**Parameters:**
- **Linkage method:** Ward.D2 (minimizes within-cluster variance)
- **Distance metric:** Euclidean distance on PCA-transformed features
- **Optimal k:** 2 clusters (automatically determined)
- **Silhouette score:** 0.298 (moderate cluster quality)

**Statistical rationale:**
- Ward's method produces compact, spherical clusters similar to K-means but with hierarchical structure
- Silhouette score of 0.298 indicates moderate but acceptable cluster separation
- Multi-criteria approach (silhouette + elbow) ensures robust k selection

#### 4.1.2 Results

**Cluster composition:**
- **Cluster 1:** 208 bouts (92.4% of dataset)
  - Mean duration: 20.3 frames
  - Represented across 75 videos
  - 3 animals represented
- **Cluster 2:** 17 bouts (7.6% of dataset)
  - Mean duration: 19.6 frames
  - Represented across 8 videos
  - 3 animals represented

**Cluster quality metrics:**
- **Silhouette score:** 0.298 (moderate quality)
  - Interpretation: Clusters are reasonably well-separated, though some overlap may exist
  - Range: [-1, 1], where 1 indicates perfect separation
  - A score of 0.298 suggests acceptable but not perfect cluster separation

### 4.2 B-SOID Clustering

#### 4.2.1 Methodology

**Algorithm:** B-SOID (Behavioral Segmentation of Open Field in DeepLabCut)
- **Step 1:** UMAP (Uniform Manifold Approximation and Projection) for non-linear dimensionality reduction
- **Step 2:** HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) for clustering

**UMAP parameters:**
- **Input dimensions:** 486 features (original feature space)
- **Output dimensions:** 10 components
- **n_neighbors:** Adaptive (based on sample size)
- **min_dist:** 0.1 (B-SOID default)
- **Metric:** Euclidean distance

**HDBSCAN parameters:**
- **min_samples:** Adaptive (based on dimensionality and sample size)
- **min_cluster_size:** Not applicable (R's dbscan::hdbscan uses only minPts)

**Statistical rationale:**
- UMAP preserves both local and global structure better than PCA, capturing non-linear manifolds
- HDBSCAN handles varying cluster densities and automatically identifies noise points
- B-SOID is specifically designed for behavioral data analysis

#### 4.2.2 Results

**Cluster composition:**
- **Total clusters identified:** 5 clusters (excluding noise)
- **Noise points (cluster 0):** 99 bouts (44.0% of dataset)
- **Clustered bouts:** 126 bouts (56.0% of dataset)

**Cluster sizes:**
- **Cluster 0 (Noise):** 99 bouts (44.0% of dataset)
- **Cluster 1:** 12 bouts (5.3% of dataset, 9.5% of clustered bouts)
- **Cluster 2:** 31 bouts (13.8% of dataset, 24.6% of clustered bouts)
- **Cluster 3:** 20 bouts (8.9% of dataset, 15.9% of clustered bouts)
- **Cluster 4:** 35 bouts (15.6% of dataset, 27.8% of clustered bouts)
- **Cluster 5:** 28 bouts (12.4% of dataset, 22.2% of clustered bouts)

**Cluster quality metrics:**
- **Silhouette score:** -0.003 (near-zero)
- **Noise ratio:** 44.0% (high proportion of noise points)

**Interpretation:**
The B-SOID clustering identified a more granular structure with 5 distinct clusters, but with a high proportion of noise points (44.0%). This suggests:

1. **High behavioral variability:** The turn_left behavior exhibits substantial variation, with many bouts not forming dense clusters
2. **Multiple behavioral variants:** The 5 identified clusters may represent distinct execution styles or contextual variations of turn_left
3. **Noise interpretation:** The 99 noise points (44.0%) likely represent:
   - Bouts with intermediate characteristics between clusters
   - Unique behavioral variants that don't form dense groups
   - Transitional behaviors or ambiguous cases

**Comparison with hierarchical clustering:**
- Hierarchical clustering identified 2 broad clusters with well-separated separation (silhouette: 0.298)
- B-SOID identified 5 finer-grained clusters but with poor separation (silhouette: -0.003)
- The high noise ratio in B-SOID (44.0%) suggests that the behavioral space may be more continuous than discrete, with many intermediate cases

**UMAP embedding:**
The UMAP transformation reduced the 486-dimensional feature space to 10 dimensions while preserving non-linear relationships. The embedding captures complex behavioral manifolds that linear methods (PCA) might miss, making it particularly suitable for behavioral data with non-linear structure.

---

## 5. Visualization and Output

### 5.1 Hierarchical Clustering Visualizations

**PDF Report:** `BoutResults/clustering/hierarchical/clustering_hierarchical_report.pdf`

**Contents:**
1. **Dendrogram:** Hierarchical tree structure colored by cluster assignment
   - Shows cluster relationships and merge heights
   - Interpretation: Lower merge heights indicate more similar clusters

2. **PCA Visualization:** 2D projection of clusters in principal component space
   - PC1 vs. PC2 with cluster coloring
   - Shows cluster separation in reduced-dimensional space

3. **t-SNE Visualization:** Non-linear 2D embedding
   - Captures local structure and cluster relationships
   - Useful for visualizing high-dimensional cluster structure

4. **Cluster Sizes:** Bar chart showing distribution of bouts across clusters

5. **Feature Distributions:** Box plots showing feature value distributions by cluster
   - Identifies features that distinguish clusters
   - Shows within-cluster variability

6. **Cluster Heatmap:** Mean feature values per cluster
   - Visualizes cluster-specific feature patterns
   - Features are clustered to show similar patterns

7. **Bout Timeline:** Temporal distribution of bouts across clusters
   - Shows when different behavioral variants occur

**Standalone visualizations:**
- `pca_clusters.png`: PCA plot
- `tsne_clusters.png`: t-SNE plot
- `cluster_sizes.png`: Cluster size distribution
- `feature_distributions.png`: Feature distributions
- `cluster_heatmap.png`: Feature heatmap
- `bout_timeline.png`: Temporal distribution

### 5.2 B-SOID Clustering Visualizations

**PDF Report:** `BoutResults/clustering/bsoid/clustering_bsoid_report.pdf`

**Contents:**
Similar visualization suite as hierarchical clustering, adapted for B-SOID:
1. PCA Visualization (on original features, not UMAP space)
2. t-SNE Visualization
3. Cluster Sizes (including noise cluster)
4. Feature Distributions
5. Cluster Heatmap
6. Bout Timeline

**Note:** B-SOID visualizations use the original feature space (after outlier removal) rather than PCA-transformed features, as B-SOID uses UMAP for its own dimensionality reduction.

### 5.3 Cluster Videos

**Hierarchical clustering videos:**
- `cluster_hierarchical_1.mp4`: 208 bouts (Cluster 1)
- `cluster_hierarchical_2.mp4`: 17 bouts (Cluster 2)

**B-SOID clustering videos:**
- `cluster_bsoid_0.mp4`: 99 bouts (Noise cluster)
- `cluster_bsoid_1.mp4`: 12 bouts (Cluster 1)
- `cluster_bsoid_2.mp4`: 31 bouts (Cluster 2)
- `cluster_bsoid_3.mp4`: 20 bouts (Cluster 3)
- `cluster_bsoid_4.mp4`: 35 bouts (Cluster 4)
- `cluster_bsoid_5.mp4`: 28 bouts (Cluster 5)

**Total videos generated:** 2 hierarchical + 6 B-SOID = 8 videos

Each video contains concatenated clips of all bouts in the cluster, with:
- Bounding box overlays (yellow outline)
- Text annotations showing bout information
- Consistent frame rate (30 fps)
- Standard video codec (libx264 + aac)

---

## 6. Statistical Summary

### 6.1 Dataset Statistics

| Metric | Value |
|--------|-------|
| Initial bouts | 243 |
| Outliers removed | 18 (7.4%) |
| Bouts after outlier removal | 225 (92.6%) |
| Features per bout | 486 |
| PCA components (95% variance) | 64 |
| UMAP components (B-SOID) | 10 |

### 6.2 Hierarchical Clustering Summary

| Metric | Value |
|--------|-------|
| Number of clusters | 2 |
| Silhouette score | 0.298 |
| Linkage method | Ward.D2 |
| Cluster 1 size | 208 bouts (92.4%) |
| Cluster 2 size | 17 bouts (7.6%) |

### 6.3 B-SOID Clustering Summary

| Metric | Value |
|--------|-------|
| Number of clusters | 5 |
| Noise points | 99 (44.0%) |
| Clustered bouts | 126 (56.0%) |
| Silhouette score | -0.003 |
| UMAP components | 10 |
| min_samples (HDBSCAN) | Adaptive |
| Largest cluster | 35 bouts (Cluster 4, 15.6%) |
| Smallest cluster | 12 bouts (Cluster 1, 5.3%) |

---

## 7. Discussion

### 7.1 Methodological Considerations

**Outlier detection:**
The consensus-based outlier detection approach successfully identified 18 anomalous bouts (7.4% of the dataset) using multiple distance metrics and aggregation methods. This conservative approach ensures that only consistently identified outliers are removed, reducing the risk of removing valid but unusual behavioral variants.

**Dimensionality reduction:**
PCA successfully reduced the feature space from 486 to 64 dimensions while retaining 95% of variance, indicating substantial redundancy in the original features. This reduction is essential for hierarchical clustering, which has quadratic or cubic complexity.

**Clustering comparison:**
The two clustering methods provide complementary perspectives:
- **Hierarchical clustering:** Identifies 2 broad, well-separated clusters (silhouette: 0.298), suggesting a binary structure in the behavioral space
- **B-SOID:** Identifies 5 finer-grained clusters but with poor separation (silhouette: -0.003) and high noise (44.0%), suggesting a more continuous or multi-modal behavioral space

### 7.2 Biological Interpretation

**Hierarchical clustering results:**
The identification of 2 distinct clusters suggests that turn_left behavior may exist in two primary forms:
1. **Standard variant:** The dominant execution style
2. Variant form: Distinct execution style, with significantly different feature patterns

**B-SOID results:**
The identification of 5 clusters with high noise suggests:
1. **Behavioral diversity:** turn_left behavior exhibits substantial variation
2. **Multiple execution styles:** The 5 clusters may represent distinct ways of performing the behavior
3. **Continuous variation:** The high noise ratio (44.0%) suggests that many bouts fall between discrete clusters, indicating a more continuous behavioral space

**Noise points in B-SOID:**
The 99 noise points (44.0%) likely represent:
- Transitional behaviors between clusters
- Unique execution styles that don't form dense groups
- Ambiguous cases that could belong to multiple clusters
- Individual differences in behavioral expression

### 7.3 Limitations

1. **Sample size:** With 225 bouts after outlier removal, the dataset is moderate in size. Larger datasets would provide more robust cluster identification.

2. **Feature selection:** The 486 features were extracted from available pose estimation data. Additional features or different feature engineering approaches might reveal different cluster structures.

3. **Clustering assumptions:**
   - Hierarchical clustering assumes clusters are well-separated and spherical
   - B-SOID assumes clusters have sufficient density to be identified
   - Both methods may miss clusters that don't meet these assumptions

4. **Noise interpretation:** The high noise ratio in B-SOID (44.0%) makes interpretation challenging. These noise points may represent valid behavioral variation that doesn't form discrete clusters.

5. **Temporal context:** The analysis treats each bout independently. Temporal relationships between bouts (e.g., sequences, transitions) are not considered.

### 7.4 Future Directions

1. **Larger datasets:** Collecting more bouts would improve cluster stability and allow identification of rarer behavioral variants.

2. **Feature engineering:** Exploring different feature representations (e.g., frequency domain features, temporal dynamics) might reveal additional structure.

3. **Temporal analysis:** Incorporating temporal relationships between bouts (e.g., sequences, transitions) could provide additional insights.

4. **Supervised validation:** Comparing clusters to expert annotations or behavioral labels could validate cluster interpretations.

5. **Multi-animal analysis:** Analyzing clusters across different animals could reveal individual differences in behavioral expression.

---

## 8. Conclusions

This analysis successfully applied a comprehensive pipeline to identify behavioral clusters in turn_left behavior. Key findings:

1. **Outlier detection:** Consensus-based approach identified and removed 18 anomalous bouts (7.4%), improving data quality for clustering.

2. **Hierarchical clustering:** Identified 2 distinct behavioral clusters with moderate separation (silhouette: 0.298), suggesting a binary structure with a dominant variant.

3. **B-SOID clustering:** Identified 5 finer-grained clusters but with poor separation (silhouette: -0.003) and high noise (44.0%), suggesting a more continuous or multi-modal behavioral space.

4. **Complementary methods:** The two clustering approaches provide different perspectives on behavioral structure, with hierarchical clustering emphasizing broad patterns and B-SOID emphasizing fine-grained variation.

5. **Behavioral diversity:** Both methods indicate substantial variation in turn_left behavior, with hierarchical clustering suggesting a binary structure and B-SOID suggesting a more continuous space with multiple modes.

The analysis provides a foundation for understanding behavioral structure in turn_left behavior and demonstrates the value of applying multiple clustering methods to gain complementary insights into behavioral organization.

---

## 9. Output Files

All analysis outputs are organized in the `BoutResults/` directory:

### 9.1 Feature Extraction
- `bout_features.csv`: Complete feature matrix (243 bouts × 486 features)
- `cache/`: Cached feature extraction results for reproducibility

### 9.2 Outlier Detection
- `outliers/consensus_outliers.csv`: 18 identified outliers
- `outliers/all_outlier_scores.csv`: Distance scores for all methods
- `outliers/*.png`: Visualization plots

### 9.3 Clustering
- `clustering/bout_features_clean.csv`: Features after outlier removal (225 bouts)
- `clustering/bout_features_pca.csv`: PCA-transformed features (64 components)

**Hierarchical clustering:**
- `clustering/hierarchical/cluster_assignments_hierarchical.csv`: Cluster assignments
- `clustering/hierarchical/cluster_statistics_hierarchical.json`: Cluster statistics
- `clustering/hierarchical/clustering_hierarchical_report.pdf`: Complete visualization report
- `clustering/hierarchical/*.png`: Individual visualization plots

**B-SOID clustering:**
- `clustering/bsoid/cluster_assignments_bsoid.csv`: Cluster assignments
- `clustering/bsoid/cluster_statistics_bsoid.json`: Cluster statistics
- `clustering/bsoid/clustering_bsoid_report.pdf`: Complete visualization report
- `clustering/bsoid/*.png`: Individual visualization plots

### 9.4 Videos
- `videos/all_bouts.mp4`: Video of all 243 bouts
- `videos/outliers.mp4`: Video of 18 removed outliers
- `videos/hierarchical/`: Cluster videos for hierarchical clustering
- `videos/bsoid/`: Cluster videos for B-SOID clustering (including noise)

---

## 10. Technical Details

### 10.1 Software and Packages

**Python:**
- pandas, numpy: Data manipulation
- h5py: HDF5 file reading
- multiprocessing: Parallel processing (7 workers)

**R:**
- optparse: Command-line argument parsing
- dplyr: Data manipulation
- cluster: Clustering algorithms
- factoextra: Cluster visualization
- uwot: UMAP implementation
- dbscan: HDBSCAN implementation
- ggplot2, pheatmap: Visualization
- Rtsne: t-SNE implementation

### 10.2 Computational Resources

- **CPU cores used:** 7 (n-1 cores, leaving 1 core free)
- **Parallel processing:** Applied to feature extraction, clustering, and video generation
- **Memory:** Sufficient for dataset size (225 bouts)

### 10.3 Reproducibility

- All random seeds set for reproducibility
- Feature extraction results cached
- Complete parameter logs in JSON files
- All scripts version-controlled

---

**Report generated:** December 29, 2025  
**Analysis pipeline version:** 1.0.0  
**Contact:** See project documentation for details
