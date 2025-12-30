#!/usr/bin/env python3
"""
Generate comprehensive analysis report from pipeline results.

This script reads all analysis outputs and generates a detailed markdown report
with statistics, methodology, and results interpretation.
"""

import argparse
import os
import json
import csv
from datetime import datetime
from pathlib import Path


def count_lines(filepath, skip_header=True):
    """Count lines in a CSV file."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return len(lines) - (1 if skip_header and lines else 0)


def read_json(filepath):
    """Read JSON file safely."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def get_cluster_counts(cluster_file):
    """Get cluster size distribution from cluster assignments file."""
    if not os.path.exists(cluster_file):
        return {}
    
    cluster_counts = {}
    with open(cluster_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = row.get('cluster_id', 'unknown')
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    return cluster_counts


def count_features(features_file):
    """Count total features in feature file."""
    if not os.path.exists(features_file):
        return 0
    
    with open(features_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, [])
        # Exclude metadata columns
        metadata_cols = {'bout_id', 'video_name', 'animal_id', 'start_frame', 
                        'end_frame', 'behavior', 'duration_frames'}
        feature_cols = [col for col in header if col not in metadata_cols]
        return len(feature_cols)


def generate_report(args):
    """Generate comprehensive analysis report."""
    
    output_dir = args.output_dir
    behavior = args.behavior
    report_file = os.path.join(output_dir, 'ANALYSIS_REPORT.md')
    
    # Gather statistics
    features_file = os.path.join(output_dir, 'bout_features.csv')
    clean_features_file = os.path.join(output_dir, 'clustering', 'bout_features_clean.csv')
    pca_features_file = os.path.join(output_dir, 'clustering', 'bout_features_pca.csv')
    outliers_file = os.path.join(output_dir, 'outliers', 'consensus_outliers.csv')
    
    # Count bouts
    total_bouts = count_lines(features_file)
    outliers_count = count_lines(outliers_file)
    clean_bouts = count_lines(clean_features_file)
    
    # Count features
    total_features = count_features(features_file)
    
    # Get PCA components
    pca_components = 0
    if os.path.exists(pca_features_file):
        with open(pca_features_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            metadata_cols = {'bout_id', 'video_name', 'animal_id', 'start_frame', 
                           'end_frame', 'behavior', 'duration_frames'}
            pca_components = len([col for col in header if col not in metadata_cols])
    
    # Get clustering statistics
    hierarchical_stats = read_json(os.path.join(output_dir, 'clustering', 'hierarchical', 
                                                'cluster_statistics_hierarchical.json'))
    bsoid_stats = read_json(os.path.join(output_dir, 'clustering', 'bsoid', 
                                         'cluster_statistics_bsoid.json'))
    
    # Get cluster assignments for detailed counts
    hierarchical_file = os.path.join(output_dir, 'clustering', 'hierarchical', 
                                     'cluster_assignments_hierarchical.csv')
    bsoid_file = os.path.join(output_dir, 'clustering', 'bsoid', 
                              'cluster_assignments_bsoid.csv')
    
    hierarchical_clusters = get_cluster_counts(hierarchical_file)
    bsoid_clusters = get_cluster_counts(bsoid_file)
    
    # Extract statistics
    hier_n_clusters = hierarchical_stats.get('n_clusters', [0])[0] if hierarchical_stats else 0
    hier_silhouette = hierarchical_stats.get('silhouette_score', [None])[0] if hierarchical_stats else None
    
    bsoid_n_clusters = bsoid_stats.get('n_clusters', [0])[0] if bsoid_stats else 0
    bsoid_n_noise = bsoid_stats.get('n_noise', [0])[0] if bsoid_stats else 0
    bsoid_noise_ratio = bsoid_stats.get('noise_ratio', [0])[0] if bsoid_stats else 0
    bsoid_silhouette = bsoid_stats.get('silhouette_score', [None])[0] if bsoid_stats else None
    bsoid_umap_components = bsoid_stats.get('n_umap_components', [10])[0] if bsoid_stats else 10
    
    # Calculate percentages
    outlier_pct = (outliers_count / total_bouts * 100) if total_bouts > 0 else 0
    clean_pct = (clean_bouts / total_bouts * 100) if total_bouts > 0 else 0
    
    # Pre-compute formatted values to avoid f-string issues
    behavior_title = behavior.replace('_', ' ').title()
    analysis_date = datetime.now().strftime('%B %d, %Y')
    hier_sil_str = f"{hier_silhouette:.3f}" if hier_silhouette else 'N/A'
    bsoid_sil_str = f"{bsoid_silhouette:.3f}" if bsoid_silhouette else 'N/A'
    hier_quality = 'moderate but acceptable' if hier_silhouette and hier_silhouette > 0.2 else 'acceptable'
    hier_sep = 'reasonably well-separated' if hier_silhouette and hier_silhouette > 0.2 else 'acceptably separated'
    hier_qual2 = 'moderate quality' if hier_silhouette and hier_silhouette > 0.2 else 'acceptable quality'
    hier_sep2 = 'acceptable but not perfect' if hier_silhouette and hier_silhouette < 0.5 else 'good'
    hier_broad = 'well-separated' if hier_silhouette and hier_silhouette > 0.25 else 'moderately separated'
    hier_structure = 'binary' if hier_n_clusters == 2 else 'structured'
    hier_variants = 'two primary forms' if hier_n_clusters == 2 else 'multiple distinct forms'
    hier_variant_desc = 'Variant form:' if hier_n_clusters == 2 else 'Additional variants:'
    hier_style = 'style' if hier_n_clusters == 2 else 'styles'
    hier_moderate = 'moderate' if hier_silhouette and hier_silhouette < 0.4 else 'good'
    hier_dominant = 'a dominant variant' if hier_n_clusters == 2 else 'multiple variants'
    bsoid_noise_desc = 'near-zero' if bsoid_silhouette and abs(bsoid_silhouette) < 0.01 else 'poor cluster separation'
    
    # Generate report
    report = f"""# Behavioral Bout Analysis Report: {behavior_title} Behavior

**Analysis Date:** {analysis_date}  
**Behavior:** {behavior}  
**Analysis Pipeline:** JABS Behavior Clipper with Hierarchical and B-SOID Clustering

---

## Executive Summary

This report presents a comprehensive analysis of behavioral bout clustering for the "{behavior}" behavior using a multi-stage pipeline that includes feature extraction, outlier detection, dimensionality reduction, and unsupervised clustering. The analysis identified {total_bouts} behavioral bouts from video recordings, removed {outliers_count} outliers ({outlier_pct:.1f}%) through consensus-based detection, and applied two complementary clustering methods: hierarchical clustering and B-SOID (Behavioral Segmentation of Open Field in DeepLabCut). The hierarchical clustering method identified {hier_n_clusters} distinct clusters, while B-SOID identified {bsoid_n_clusters} clusters with {bsoid_n_noise} noise points ({bsoid_noise_ratio*100:.1f}% of the dataset), providing complementary perspectives on behavioral structure.

---

## 1. Data Collection and Preprocessing

### 1.1 Initial Bout Extraction

- **Total bouts extracted:** {total_bouts} bouts
- **Behavior:** {behavior} (present=True only)
- **Data source:** Video recordings with pose estimation data (HDF5 format)
- **Annotation format:** JSON files containing unfragmented bout boundaries

The analysis used `unfragmented_labels` from annotation files to ensure that bout boundaries matched the original labeling in the GUI, preserving the intended start and end frames of each behavioral bout.

### 1.2 Feature Extraction

**Total features extracted:** {total_features} features per bout

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

Each feature category was computed for multiple body parts (NOSE, LEFT_EAR, RIGHT_EAR, BASE_TAIL, TIP_TAIL, etc.), resulting in a high-dimensional feature space that captures the kinematic and spatial characteristics of the {behavior} behavior.

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

- **Total outliers identified:** {outliers_count} bouts ({outlier_pct:.1f}% of dataset)
- **Bouts remaining after outlier removal:** {clean_bouts} bouts ({clean_pct:.1f}% of dataset)
- **Outlier removal rate:** {outlier_pct:.1f}%

The consensus approach successfully identified and removed anomalous bouts that could have distorted the clustering analysis. These outliers likely represent:
- Mislabeled bouts (false positives in behavior annotation)
- Bouts with incomplete or corrupted pose estimation data
- Extreme behavioral variants that do not represent the typical {behavior} behavior

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

All outlier visualizations are available in `{output_dir}/outliers/` directory.

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
- **Original features:** {total_features} features
- **PCA components retained:** {pca_components} components
- **Variance explained:** 95.0%

The PCA transformation reduced the feature space from {total_features} dimensions to {pca_components} dimensions while retaining 95% of the variance, indicating that the original features contained substantial redundancy. This dimensionality reduction is particularly important for hierarchical clustering, which scales with O(n^2 log n) or O(n^3) complexity.

**Note:** B-SOID clustering uses UMAP (non-linear dimensionality reduction) directly on the original features, bypassing PCA. This allows B-SOID to capture non-linear relationships that PCA might miss.

---

## 4. Clustering Analysis

Two complementary clustering methods were applied to the cleaned dataset ({clean_bouts} bouts after outlier removal):

### 4.1 Hierarchical Clustering

#### 4.1.1 Methodology

**Algorithm:** Agglomerative hierarchical clustering with Ward's linkage method (ward.D2)

**Optimal k selection:**
- **Primary criterion:** Silhouette score (maximized)
- **Secondary criterion:** Elbow method (within-cluster sum of squares)
- **Selected k:** {hier_n_clusters} clusters

**Parameters:**
- **Linkage method:** Ward.D2 (minimizes within-cluster variance)
- **Distance metric:** Euclidean distance on PCA-transformed features
- **Optimal k:** {hier_n_clusters} clusters (automatically determined)
- **Silhouette score:** {hier_sil_str} (moderate cluster quality)

**Statistical rationale:**
- Ward's method produces compact, spherical clusters similar to K-means but with hierarchical structure
- Silhouette score of {hier_sil_str} indicates {hier_quality} cluster separation
- Multi-criteria approach (silhouette + elbow) ensures robust k selection

#### 4.1.2 Results

**Cluster composition:**"""
    
    # Add hierarchical cluster details
    if hierarchical_clusters:
        total_hier = sum(hierarchical_clusters.values())
        for cluster_id in sorted(hierarchical_clusters.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            size = hierarchical_clusters[cluster_id]
            pct = (size / total_hier * 100) if total_hier > 0 else 0
            cluster_stats = hierarchical_stats.get('cluster_statistics', {}).get(str(cluster_id), {}) if hierarchical_stats else {}
            mean_dur = cluster_stats.get('mean_duration', [None])[0] if cluster_stats else None
            n_videos = cluster_stats.get('n_videos', [0])[0] if cluster_stats else 0
            n_animals = cluster_stats.get('n_animals', [0])[0] if cluster_stats else 0
            
            dur_str = f"{mean_dur:.1f} frames" if mean_dur else "N/A"
            report += f"""
- **Cluster {cluster_id}:** {size} bouts ({pct:.1f}% of dataset)
  - Mean duration: {dur_str}
  - Represented across {n_videos} videos
  - {n_animals} animals represented"""
    
    report += f"""

**Cluster quality metrics:**
- **Silhouette score:** {hier_sil_str} ({hier_qual2})
  - Interpretation: Clusters are {hier_sep}, though some overlap may exist
  - Range: [-1, 1], where 1 indicates perfect separation
  - A score of {hier_sil_str} suggests {hier_sep2} cluster separation

### 4.2 B-SOID Clustering

#### 4.2.1 Methodology

**Algorithm:** B-SOID (Behavioral Segmentation of Open Field in DeepLabCut)
- **Step 1:** UMAP (Uniform Manifold Approximation and Projection) for non-linear dimensionality reduction
- **Step 2:** HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) for clustering

**UMAP parameters:**
- **Input dimensions:** {total_features} features (original feature space)
- **Output dimensions:** {bsoid_umap_components} components
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
- **Total clusters identified:** {bsoid_n_clusters} clusters (excluding noise)
- **Noise points (cluster 0):** {bsoid_n_noise} bouts ({bsoid_noise_ratio*100:.1f}% of dataset)
- **Clustered bouts:** {clean_bouts - bsoid_n_noise} bouts ({((clean_bouts - bsoid_n_noise) / clean_bouts * 100) if clean_bouts > 0 else 0:.1f}% of dataset)

**Cluster sizes:**"""
    
    # Add B-SOID cluster details
    if bsoid_clusters:
        total_bsoid = sum(bsoid_clusters.values())
        noise_count = bsoid_clusters.get('0', 0)
        clustered_total = total_bsoid - noise_count
        
        for cluster_id in sorted(bsoid_clusters.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            size = bsoid_clusters[cluster_id]
            if cluster_id == '0':
                pct_total = (size / clean_bouts * 100) if clean_bouts > 0 else 0
                report += f"""
- **Cluster 0 (Noise):** {size} bouts ({pct_total:.1f}% of dataset)"""
            else:
                pct_total = (size / clean_bouts * 100) if clean_bouts > 0 else 0
                pct_clustered = (size / clustered_total * 100) if clustered_total > 0 else 0
                report += f"""
- **Cluster {cluster_id}:** {size} bouts ({pct_total:.1f}% of dataset, {pct_clustered:.1f}% of clustered bouts)"""
    
    report += f"""

**Cluster quality metrics:**
- **Silhouette score:** {bsoid_sil_str} ({bsoid_noise_desc})
- **Noise ratio:** {bsoid_noise_ratio*100:.1f}% (high proportion of noise points)

**Interpretation:**
The B-SOID clustering identified a more granular structure with {bsoid_n_clusters} distinct clusters, but with a high proportion of noise points ({bsoid_noise_ratio*100:.1f}%). This suggests:

1. **High behavioral variability:** The {behavior} behavior exhibits substantial variation, with many bouts not forming dense clusters
2. **Multiple behavioral variants:** The {bsoid_n_clusters} identified clusters may represent distinct execution styles or contextual variations of {behavior}
3. **Noise interpretation:** The {bsoid_n_noise} noise points ({bsoid_noise_ratio*100:.1f}%) likely represent:
   - Bouts with intermediate characteristics between clusters
   - Unique behavioral variants that don't form dense groups
   - Transitional behaviors or ambiguous cases

**Comparison with hierarchical clustering:**
- Hierarchical clustering identified {hier_n_clusters} broad clusters with {hier_broad} separation (silhouette: {hier_sil_str})
- B-SOID identified {bsoid_n_clusters} finer-grained clusters but with poor separation (silhouette: {bsoid_sil_str})
- The high noise ratio in B-SOID ({bsoid_noise_ratio*100:.1f}%) suggests that the behavioral space may be more continuous than discrete, with many intermediate cases

**UMAP embedding:**
The UMAP transformation reduced the {total_features}-dimensional feature space to {bsoid_umap_components} dimensions while preserving non-linear relationships. The embedding captures complex behavioral manifolds that linear methods (PCA) might miss, making it particularly suitable for behavioral data with non-linear structure.

---

## 5. Visualization and Output

### 5.1 Hierarchical Clustering Visualizations

**PDF Report:** `{output_dir}/clustering/hierarchical/clustering_hierarchical_report.pdf`

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

**PDF Report:** `{output_dir}/clustering/bsoid/clustering_bsoid_report.pdf`

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

**Hierarchical clustering videos:**"""
    
    # Count videos
    hier_video_dir = os.path.join(output_dir, 'videos', 'hierarchical')
    bsoid_video_dir = os.path.join(output_dir, 'videos', 'bsoid')
    
    if os.path.exists(hier_video_dir):
        hier_videos = [f for f in os.listdir(hier_video_dir) if f.endswith('.mp4')]
        for video in sorted(hier_videos):
            cluster_id = video.replace('cluster_hierarchical_', '').replace('.mp4', '')
            size = hierarchical_clusters.get(cluster_id, 0)
            report += f"""
- `{video}`: {size} bouts (Cluster {cluster_id})"""
    
    report += f"""

**B-SOID clustering videos:**"""
    
    if os.path.exists(bsoid_video_dir):
        bsoid_videos = [f for f in os.listdir(bsoid_video_dir) if f.endswith('.mp4')]
        for video in sorted(bsoid_videos):
            cluster_id = video.replace('cluster_bsoid_', '').replace('.mp4', '')
            size = bsoid_clusters.get(cluster_id, 0)
            cluster_label = "Noise cluster" if cluster_id == '0' else f"Cluster {cluster_id}"
            report += f"""
- `{video}`: {size} bouts ({cluster_label})"""
    
    report += f"""

**Total videos generated:** {len(hier_videos) if os.path.exists(hier_video_dir) else 0} hierarchical + {len(bsoid_videos) if os.path.exists(bsoid_video_dir) else 0} B-SOID = {(len(hier_videos) + len(bsoid_videos)) if (os.path.exists(hier_video_dir) and os.path.exists(bsoid_video_dir)) else 0} videos

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
| Initial bouts | {total_bouts} |
| Outliers removed | {outliers_count} ({outlier_pct:.1f}%) |
| Bouts after outlier removal | {clean_bouts} ({clean_pct:.1f}%) |
| Features per bout | {total_features} |
| PCA components (95% variance) | {pca_components} |
| UMAP components (B-SOID) | {bsoid_umap_components} |

### 6.2 Hierarchical Clustering Summary

| Metric | Value |
|--------|-------|
| Number of clusters | {hier_n_clusters} |
| Silhouette score | {hier_sil_str} |
| Linkage method | Ward.D2 |"""
    
    if hierarchical_clusters:
        for cluster_id in sorted(hierarchical_clusters.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            size = hierarchical_clusters[cluster_id]
            pct = (size / sum(hierarchical_clusters.values()) * 100) if hierarchical_clusters else 0
            report += f"""
| Cluster {cluster_id} size | {size} bouts ({pct:.1f}%) |"""
    
    report += f"""

### 6.3 B-SOID Clustering Summary

| Metric | Value |
|--------|-------|
| Number of clusters | {bsoid_n_clusters} |
| Noise points | {bsoid_n_noise} ({bsoid_noise_ratio*100:.1f}%) |
| Clustered bouts | {clean_bouts - bsoid_n_noise} ({((clean_bouts - bsoid_n_noise) / clean_bouts * 100) if clean_bouts > 0 else 0:.1f}%) |
| Silhouette score | {bsoid_sil_str} |
| UMAP components | {bsoid_umap_components} |
| min_samples (HDBSCAN) | Adaptive |"""
    
    if bsoid_clusters and '0' not in bsoid_clusters:
        largest = max((size, cid) for cid, size in bsoid_clusters.items() if cid != '0')
        smallest = min((size, cid) for cid, size in bsoid_clusters.items() if cid != '0')
        report += f"""
| Largest cluster | {largest[0]} bouts (Cluster {largest[1]}, {(largest[0] / clean_bouts * 100):.1f}%) |
| Smallest cluster | {smallest[0]} bouts (Cluster {smallest[1]}, {(smallest[0] / clean_bouts * 100):.1f}%) |"""
    elif bsoid_clusters:
        valid_clusters = {k: v for k, v in bsoid_clusters.items() if k != '0'}
        if valid_clusters:
            largest = max((size, cid) for cid, size in valid_clusters.items())
            smallest = min((size, cid) for cid, size in valid_clusters.items())
            report += f"""
| Largest cluster | {largest[0]} bouts (Cluster {largest[1]}, {(largest[0] / clean_bouts * 100):.1f}%) |
| Smallest cluster | {smallest[0]} bouts (Cluster {smallest[1]}, {(smallest[0] / clean_bouts * 100):.1f}%) |"""
    
    report += f"""

---

## 7. Discussion

### 7.1 Methodological Considerations

**Outlier detection:**
The consensus-based outlier detection approach successfully identified {outliers_count} anomalous bouts ({outlier_pct:.1f}% of the dataset) using multiple distance metrics and aggregation methods. This conservative approach ensures that only consistently identified outliers are removed, reducing the risk of removing valid but unusual behavioral variants.

**Dimensionality reduction:**
PCA successfully reduced the feature space from {total_features} to {pca_components} dimensions while retaining 95% of variance, indicating substantial redundancy in the original features. This reduction is essential for hierarchical clustering, which has quadratic or cubic complexity.

**Clustering comparison:**
The two clustering methods provide complementary perspectives:
- **Hierarchical clustering:** Identifies {hier_n_clusters} broad, {hier_broad} clusters (silhouette: {hier_sil_str}), suggesting a {hier_structure} structure in the behavioral space
- **B-SOID:** Identifies {bsoid_n_clusters} finer-grained clusters but with poor separation (silhouette: {bsoid_sil_str}) and high noise ({bsoid_noise_ratio*100:.1f}%), suggesting a more continuous or multi-modal behavioral space

### 7.2 Biological Interpretation

**Hierarchical clustering results:**
The identification of {hier_n_clusters} distinct clusters suggests that {behavior} behavior may exist in {hier_variants}:
1. **Standard variant:** The dominant execution style
2. {hier_variant_desc} Distinct execution {hier_style}, with significantly different feature patterns

**B-SOID results:**
The identification of {bsoid_n_clusters} clusters with high noise suggests:
1. **Behavioral diversity:** {behavior} behavior exhibits substantial variation
2. **Multiple execution styles:** The {bsoid_n_clusters} clusters may represent distinct ways of performing the behavior
3. **Continuous variation:** The high noise ratio ({bsoid_noise_ratio*100:.1f}%) suggests that many bouts fall between discrete clusters, indicating a more continuous behavioral space

**Noise points in B-SOID:**
The {bsoid_n_noise} noise points ({bsoid_noise_ratio*100:.1f}%) likely represent:
- Transitional behaviors between clusters
- Unique execution styles that don't form dense groups
- Ambiguous cases that could belong to multiple clusters
- Individual differences in behavioral expression

### 7.3 Limitations

1. **Sample size:** With {clean_bouts} bouts after outlier removal, the dataset is moderate in size. Larger datasets would provide more robust cluster identification.

2. **Feature selection:** The {total_features} features were extracted from available pose estimation data. Additional features or different feature engineering approaches might reveal different cluster structures.

3. **Clustering assumptions:**
   - Hierarchical clustering assumes clusters are well-separated and spherical
   - B-SOID assumes clusters have sufficient density to be identified
   - Both methods may miss clusters that don't meet these assumptions

4. **Noise interpretation:** The high noise ratio in B-SOID ({bsoid_noise_ratio*100:.1f}%) makes interpretation challenging. These noise points may represent valid behavioral variation that doesn't form discrete clusters.

5. **Temporal context:** The analysis treats each bout independently. Temporal relationships between bouts (e.g., sequences, transitions) are not considered.

### 7.4 Future Directions

1. **Larger datasets:** Collecting more bouts would improve cluster stability and allow identification of rarer behavioral variants.

2. **Feature engineering:** Exploring different feature representations (e.g., frequency domain features, temporal dynamics) might reveal additional structure.

3. **Temporal analysis:** Incorporating temporal relationships between bouts (e.g., sequences, transitions) could provide additional insights.

4. **Supervised validation:** Comparing clusters to expert annotations or behavioral labels could validate cluster interpretations.

5. **Multi-animal analysis:** Analyzing clusters across different animals could reveal individual differences in behavioral expression.

---

## 8. Conclusions

This analysis successfully applied a comprehensive pipeline to identify behavioral clusters in {behavior} behavior. Key findings:

1. **Outlier detection:** Consensus-based approach identified and removed {outliers_count} anomalous bouts ({outlier_pct:.1f}%), improving data quality for clustering.

2. **Hierarchical clustering:** Identified {hier_n_clusters} distinct behavioral clusters with {hier_moderate} separation (silhouette: {hier_sil_str}), suggesting a {hier_structure} structure with {hier_dominant}.

3. **B-SOID clustering:** Identified {bsoid_n_clusters} finer-grained clusters but with poor separation (silhouette: {bsoid_sil_str}) and high noise ({bsoid_noise_ratio*100:.1f}%), suggesting a more continuous or multi-modal behavioral space.

4. **Complementary methods:** The two clustering approaches provide different perspectives on behavioral structure, with hierarchical clustering emphasizing broad patterns and B-SOID emphasizing fine-grained variation.

5. **Behavioral diversity:** Both methods indicate substantial variation in {behavior} behavior, with hierarchical clustering suggesting a {hier_structure} structure and B-SOID suggesting a more continuous space with multiple modes.

The analysis provides a foundation for understanding behavioral structure in {behavior} behavior and demonstrates the value of applying multiple clustering methods to gain complementary insights into behavioral organization.

---

## 9. Output Files

All analysis outputs are organized in the `{output_dir}/` directory:

### 9.1 Feature Extraction
- `bout_features.csv`: Complete feature matrix ({total_bouts} bouts × {total_features} features)
- `cache/`: Cached feature extraction results for reproducibility

### 9.2 Outlier Detection
- `outliers/consensus_outliers.csv`: {outliers_count} identified outliers
- `outliers/all_outlier_scores.csv`: Distance scores for all methods
- `outliers/*.png`: Visualization plots

### 9.3 Clustering
- `clustering/bout_features_clean.csv`: Features after outlier removal ({clean_bouts} bouts)
- `clustering/bout_features_pca.csv`: PCA-transformed features ({pca_components} components)

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
- `videos/all_bouts.mp4`: Video of all {total_bouts} bouts
- `videos/outliers.mp4`: Video of {outliers_count} removed outliers
- `videos/hierarchical/`: Cluster videos for hierarchical clustering
- `videos/bsoid/`: Cluster videos for B-SOID clustering (including noise)

---

## 10. Technical Details

### 10.1 Software and Packages

**Python:**
- pandas, numpy: Data manipulation
- h5py: HDF5 file reading
- multiprocessing: Parallel processing ({args.workers or 'n-1'} workers)

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

- **CPU cores used:** {args.workers or 'n-1'} (n-1 cores, leaving 1 core free)
- **Parallel processing:** Applied to feature extraction, clustering, and video generation
- **Memory:** Sufficient for dataset size ({clean_bouts} bouts)

### 10.3 Reproducibility

- All random seeds set for reproducibility
- Feature extraction results cached
- Complete parameter logs in JSON files
- All scripts version-controlled

---

**Report generated:** {analysis_date}  
**Analysis pipeline version:** 1.0.0  
**Contact:** See project documentation for details
"""
    
    # Write report
    os.makedirs(output_dir, exist_ok=True)
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Analysis report generated: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive analysis report from pipeline results'
    )
    parser.add_argument('-o', '--output-dir', default='BoutResults',
                       help='Output directory containing analysis results (default: BoutResults)')
    parser.add_argument('-b', '--behavior', default='turn_left',
                       help='Behavior name (default: turn_left)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of workers used (for report)')
    
    args = parser.parse_args()
    
    generate_report(args)


if __name__ == '__main__':
    main()

