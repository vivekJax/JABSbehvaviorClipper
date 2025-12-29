#!/usr/bin/env Rscript
# Generate visualizations for clustering analysis
#
# This script creates comprehensive plots for clustering results:
# - PCA biplots with cluster assignments
# - Cluster assignments in PCA space
# - Dendrograms for hierarchical clustering
# - Cluster size distributions
# - Feature distributions by cluster
#
# Usage:
#   Rscript analysis_r/visualize_clusters.R --input bout_features.csv --output-dir results/clustering

# Add default library paths
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  library(factoextra)
  library(RColorBrewer)
})

# Source utility functions
source("analysis_r/utils/data_preprocessing.R")

# Parse command line arguments
option_list <- list(
  make_option(c("-i", "--input"), type="character", default="results/bout_features_filtered.csv",
              help="Input CSV file with bout features (default: results/bout_features_filtered.csv)"),
  make_option(c("-o", "--output-dir"), type="character", default="results/clustering",
              help="Output directory for visualizations (default: results/clustering)"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Load PCA results if available
load_pca_results <- function(pca_file) {
  if (file.exists(pca_file)) {
    load(pca_file)
    return(pca_result)
  }
  return(NULL)
}

# Main execution
main <- function() {
  cat("============================================================\n")
  cat("Clustering Visualization\n")
  cat("============================================================\n\n")
  
  # Create output directory and method-specific subdirectories
  dir.create(opt$`output-dir`, showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(opt$`output-dir`, "kmeans"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(opt$`output-dir`, "hierarchical"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(opt$`output-dir`, "dbscan"), showWarnings = FALSE, recursive = TRUE)
  
  # Close any open graphics devices to prevent Rplots.pdf in main directory
  while (dev.cur() != 1) {
    dev.off()
  }
  
  # Suppress default PDF device (prevents Rplots.pdf from being created)
  pdf(NULL)
  
  # Load features
  cat(sprintf("Loading features from: %s\n", opt$input))
  if (!file.exists(opt$input)) {
    stop(sprintf("Input file not found: %s", opt$input))
  }
  
  df <- read.csv(opt$input, stringsAsFactors = FALSE)
  cat(sprintf("Loaded %d bouts\n", nrow(df)))
  
  # Prepare features
  cat("\nPreprocessing features...\n")
  exclude_cols <- c("bout_id", "video_name", "animal_id", "start_frame", "end_frame", "behavior")
  prep_result <- prepare_features(df, exclude_cols = exclude_cols)
  
  processed_df <- prep_result$processed_df
  feature_cols <- prep_result$feature_cols
  
  # Extract feature matrix
  feature_matrix <- as.matrix(processed_df[, feature_cols, drop = FALSE])
  feature_matrix[!is.finite(feature_matrix)] <- 0
  
  # Load or compute PCA (PCA results are in the clustering directory)
  pca_file <- file.path(opt$`output-dir`, "pca_results.RData")
  if (file.exists(pca_file)) {
    cat("Loading PCA results from file...\n")
    load(pca_file)
    # pca_result is a list with pca_result$pca_result being the actual prcomp object
    if (is.list(pca_result) && "pca_result" %in% names(pca_result)) {
      pca_obj <- pca_result$pca_result
    } else {
      pca_obj <- pca_result
    }
  } else {
    cat("Computing PCA...\n")
    pca_obj <- prcomp(feature_matrix, center = TRUE, scale. = TRUE)
  }
  
  # Use pca_obj for all operations
  pca_result <- pca_obj
  
  # Load cluster assignments (from method-specific folders)
  cluster_files <- list(
    kmeans = file.path(opt$`output-dir`, "kmeans", "cluster_assignments_kmeans.csv"),
    hierarchical = file.path(opt$`output-dir`, "hierarchical", "cluster_assignments_hierarchical.csv"),
    dbscan = file.path(opt$`output-dir`, "dbscan", "cluster_assignments_dbscan.csv")
  )
  
  cat("\nGenerating visualizations...\n")
  
  # 1. PCA Scree Plot (shared)
  cat("  [1/11] PCA Scree Plot\n")
  png(file.path(opt$`output-dir`, "pca_scree_plot.png"), width = 1200, height = 800, res = 300)
  print(fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50), 
                 title = "PCA Scree Plot - Variance Explained by Principal Components"))
  dev.off()
  
  # 2. PCA Biplot (shared)
  cat("  [2/11] PCA Biplot\n")
  png(file.path(opt$`output-dir`, "pca_biplot.png"), width = 1200, height = 1200, res = 300)
  print(fviz_pca_biplot(pca_result, 
                       title = "PCA Biplot - First Two Principal Components",
                       geom = "point",
                       alpha.ind = 0.5))
  dev.off()
  
  # 3. K-means cluster visualization
  if (file.exists(cluster_files$kmeans)) {
    cat("  [3/11] K-means Clusters in PCA Space\n")
    clusters_km <- read.csv(cluster_files$kmeans, stringsAsFactors = FALSE)
    clusters_km <- clusters_km[match(processed_df$bout_id, clusters_km$bout_id), ]
    
    pca_df_km <- data.frame(
      PC1 = pca_result$x[, 1],
      PC2 = pca_result$x[, 2],
      cluster = as.factor(clusters_km$cluster)
    )
    p <- ggplot(pca_df_km, aes(x = PC1, y = PC2, color = cluster)) +
      geom_point(alpha = 0.6, size = 1) +
      scale_color_brewer(palette = "Set2") +
      labs(title = "K-means Clusters in PCA Space",
           x = "Principal Component 1",
           y = "Principal Component 2") +
      theme_minimal()
    ggsave(file.path(opt$`output-dir`, "kmeans", "kmeans_clusters_pca.png"), p, 
           width = 12, height = 10, dpi = 300)
    
    # Cluster size bar plot
    cat("  [4/11] K-means Cluster Sizes\n")
    cluster_sizes <- table(clusters_km$cluster)
    cluster_size_df <- data.frame(
      cluster = names(cluster_sizes),
      count = as.numeric(cluster_sizes)
    )
    
    p <- ggplot(cluster_size_df, aes(x = cluster, y = count, fill = cluster)) +
      geom_bar(stat = "identity") +
      scale_fill_brewer(palette = "Set2") +
      labs(title = "K-means Cluster Sizes",
           x = "Cluster ID",
           y = "Number of Bouts") +
      theme_minimal() +
      theme(legend.position = "none")
    
    ggsave(file.path(opt$`output-dir`, "kmeans", "kmeans_cluster_sizes.png"), p, 
           width = 8, height = 6, dpi = 300)
  }
  
  # 4. Hierarchical clustering dendrogram
  if (file.exists(cluster_files$hierarchical)) {
    cat("  [5/11] Hierarchical Clustering Dendrogram\n")
    clusters_hier <- read.csv(cluster_files$hierarchical, stringsAsFactors = FALSE)
    clusters_hier <- clusters_hier[match(processed_df$bout_id, clusters_hier$bout_id), ]
    
    # Get unique clusters sorted by ID to ensure consistent color mapping
    unique_clusters <- sort(unique(clusters_hier$cluster))
    n_clusters <- length(unique_clusters)
    
    # Create consistent color palette based on sorted cluster IDs
    # Use RColorBrewer Set2 palette, mapped to sorted cluster IDs
    base_colors <- brewer.pal(max(3, min(8, n_clusters)), "Set2")
    if (n_clusters > length(base_colors)) {
      # If more clusters than colors, interpolate
      cluster_colors <- colorRampPalette(base_colors)(n_clusters)
    } else {
      cluster_colors <- base_colors[1:n_clusters]
    }
    # Create named vector mapping cluster ID to color (sorted by cluster ID)
    names(cluster_colors) <- as.character(unique_clusters)
    
    # Compute distance and hierarchical clustering for dendrogram
    dist_matrix <- dist(pca_result$x[, 1:min(50, ncol(pca_result$x))], method = "euclidean")
    hc <- hclust(dist_matrix, method = "ward.D2")
    
    # Cut tree to get dendrogram cluster assignments (these are 1..k)
    dend_clusters <- cutree(hc, k = n_clusters)
    
    # Map dendrogram cluster numbers (1..k) to actual cluster IDs
    # For each dendrogram cluster number, find the most common actual cluster ID
    dend_to_actual <- numeric(max(dend_clusters))
    for (dend_id in 1:max(dend_clusters)) {
      indices <- which(dend_clusters == dend_id)
      if (length(indices) > 0) {
        actual_ids <- clusters_hier$cluster[indices]
        most_common <- as.numeric(names(sort(table(actual_ids), decreasing = TRUE))[1])
        dend_to_actual[dend_id] <- most_common
      }
    }
    
    # fviz_dend assigns colors to clusters 1..k in dendrogram order
    # We need to provide colors in the order that matches how fviz_dend assigns them
    # fviz_dend will assign color[1] to dendrogram cluster 1, color[2] to dendrogram cluster 2, etc.
    # So we need colors ordered by dendrogram cluster number, but mapped to actual cluster IDs
    dend_color_vector <- cluster_colors[as.character(dend_to_actual[1:max(dend_clusters)])]
    
    # Plot dendrogram with consistent colors
    png(file.path(opt$`output-dir`, "hierarchical", "hierarchical_dendrogram.png"), width = 1600, height = 800, res = 300)
    print(fviz_dend(hc, k = n_clusters, 
                   cex = 0.5,
                   k_colors = dend_color_vector,
                   main = "Hierarchical Clustering Dendrogram"))
    dev.off()
    
    # Hierarchical clusters in PCA space - use the same color mapping
    cat("  [6/11] Hierarchical Clusters in PCA Space\n")
    pca_df <- data.frame(
      PC1 = pca_result$x[, 1],
      PC2 = pca_result$x[, 2],
      cluster = as.factor(clusters_hier$cluster)
    )
    p <- ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster)) +
      geom_point(alpha = 0.6, size = 1) +
      scale_color_manual(values = cluster_colors) +
      labs(title = "Hierarchical Clusters in PCA Space",
           x = "Principal Component 1",
           y = "Principal Component 2") +
      theme_minimal()
    
    ggsave(file.path(opt$`output-dir`, "hierarchical", "hierarchical_clusters_pca.png"), p, 
           width = 12, height = 10, dpi = 300)
    
    # Feature importance for each cluster
    cat("  [7/11] Hierarchical Cluster Feature Importance\n")
    
    # Calculate feature means for each cluster and overall
    feature_matrix_clustered <- feature_matrix
    cluster_assignments <- clusters_hier$cluster
    
    # Overall feature means and standard deviations
    overall_means <- colMeans(feature_matrix_clustered, na.rm = TRUE)
    overall_sds <- apply(feature_matrix_clustered, 2, sd, na.rm = TRUE)
    overall_sds[overall_sds == 0] <- 1  # Avoid division by zero
    
    # Calculate feature importance for each cluster
    n_top_features <- 15  # Number of top features to show per cluster
    
    for (cluster_id in unique_clusters) {
      cluster_indices <- which(cluster_assignments == cluster_id)
      n_cluster_bouts <- length(cluster_indices)
      
      if (n_cluster_bouts < 2) {
        cat(sprintf("    Skipping cluster %d (only %d bout)\n", cluster_id, n_cluster_bouts))
        next
      }
      
      # Cluster feature means
      cluster_means <- colMeans(feature_matrix_clustered[cluster_indices, , drop = FALSE], na.rm = TRUE)
      
      # Calculate z-scores (standardized difference from overall mean)
      z_scores <- (cluster_means - overall_means) / overall_sds
      
      # Calculate fold change (ratio to overall mean, avoiding division by zero)
      fold_changes <- cluster_means / (overall_means + 1e-10)
      
      # Create importance score: absolute z-score weighted by fold change direction
      importance_scores <- abs(z_scores) * sign(fold_changes - 1)
      
      # Get top features (highest absolute z-scores)
      top_indices <- order(abs(z_scores), decreasing = TRUE)[1:min(n_top_features, length(z_scores))]
      top_features <- names(z_scores)[top_indices]
      top_z_scores <- z_scores[top_indices]
      top_fold_changes <- fold_changes[top_indices]
      
      # Create data frame for plotting
      importance_df <- data.frame(
        feature = top_features,
        z_score = top_z_scores,
        fold_change = top_fold_changes,
        cluster_mean = cluster_means[top_indices],
        overall_mean = overall_means[top_indices],
        stringsAsFactors = FALSE
      )
      
      # Sort by absolute z-score
      importance_df <- importance_df[order(abs(importance_df$z_score), decreasing = TRUE), ]
      
      # Create bar plot of top features
      p <- ggplot(importance_df, aes(x = reorder(feature, abs(z_score)), y = z_score, fill = z_score > 0)) +
        geom_bar(stat = "identity") +
        scale_fill_manual(values = c("FALSE" = "#E74C3C", "TRUE" = "#3498DB"),
                         labels = c("Below Average", "Above Average"),
                         name = "Direction") +
        coord_flip() +
        labs(title = sprintf("Top Features for Hierarchical Cluster %d (n=%d bouts)", cluster_id, n_cluster_bouts),
             subtitle = "Z-scores: standardized difference from overall mean",
             x = "Feature",
             y = "Z-Score (Standardized Difference)") +
        theme_minimal() +
        theme(axis.text.y = element_text(size = 8),
              plot.title = element_text(size = 12, face = "bold"),
              plot.subtitle = element_text(size = 10))
      
      ggsave(file.path(opt$`output-dir`, "hierarchical", sprintf("hierarchical_cluster_%d_features.png", cluster_id)), 
             p, width = 14, height = 10, dpi = 300)
      
      # Create heatmap comparing cluster means to overall means for top features
      heatmap_data <- data.frame(
        feature = rep(importance_df$feature, 2),
        value = c(importance_df$cluster_mean, importance_df$overall_mean),
        type = rep(c("Cluster Mean", "Overall Mean"), each = nrow(importance_df))
      )
      
      # Normalize for better visualization (z-score within each feature)
      for (feat in unique(heatmap_data$feature)) {
        feat_indices <- heatmap_data$feature == feat
        feat_values <- heatmap_data$value[feat_indices]
        if (sd(feat_values, na.rm = TRUE) > 0) {
          heatmap_data$value[feat_indices] <- (feat_values - mean(feat_values, na.rm = TRUE)) / sd(feat_values, na.rm = TRUE)
        }
      }
      
      # Create ordering based on z-scores
      feature_order <- importance_df$feature[order(abs(importance_df$z_score), decreasing = TRUE)]
      heatmap_data$feature <- factor(heatmap_data$feature, levels = feature_order)
      
      p_heatmap <- ggplot(heatmap_data, aes(x = type, y = feature, fill = value)) +
        geom_tile() +
        scale_fill_gradient2(low = "#E74C3C", mid = "white", high = "#3498DB",
                            midpoint = 0, name = "Normalized\nValue") +
        labs(title = sprintf("Feature Comparison: Cluster %d vs Overall Mean", cluster_id),
             subtitle = "Normalized values for comparison",
             x = "",
             y = "Feature") +
        theme_minimal() +
        theme(axis.text.y = element_text(size = 8),
              plot.title = element_text(size = 12, face = "bold"),
              plot.subtitle = element_text(size = 10))
      
      ggsave(file.path(opt$`output-dir`, "hierarchical", sprintf("hierarchical_cluster_%d_heatmap.png", cluster_id)), 
             p_heatmap, width = 8, height = 10, dpi = 300)
    }
    
    # Create combined feature importance heatmap across all clusters
    cat("  [8/11] Hierarchical Cluster Feature Importance Heatmap (All Clusters)\n")
    
    # Calculate z-scores for all clusters
    all_cluster_means <- matrix(NA, nrow = length(unique_clusters), ncol = length(feature_cols))
    rownames(all_cluster_means) <- as.character(unique_clusters)
    colnames(all_cluster_means) <- feature_cols
    
    for (cluster_id in unique_clusters) {
      cluster_indices <- which(cluster_assignments == cluster_id)
      if (length(cluster_indices) >= 2) {
        all_cluster_means[as.character(cluster_id), ] <- colMeans(feature_matrix_clustered[cluster_indices, , drop = FALSE], na.rm = TRUE)
      }
    }
    
    # Calculate z-scores for all clusters
    all_z_scores <- (all_cluster_means - matrix(overall_means, nrow = nrow(all_cluster_means), 
                                                 ncol = ncol(all_cluster_means), byrow = TRUE)) / 
                     matrix(overall_sds, nrow = nrow(all_cluster_means), 
                            ncol = ncol(all_cluster_means), byrow = TRUE)
    
    # Find features with highest variance across clusters (most distinguishing)
    feature_variance <- apply(all_z_scores, 2, function(x) var(x, na.rm = TRUE))
    top_distinguishing_features <- names(sort(feature_variance, decreasing = TRUE))[1:min(30, length(feature_variance))]
    
    # Create heatmap data
    heatmap_df <- expand.grid(cluster = as.character(unique_clusters), 
                             feature = top_distinguishing_features)
    heatmap_df$z_score <- NA
    
    for (i in 1:nrow(heatmap_df)) {
      clust <- heatmap_df$cluster[i]
      feat <- heatmap_df$feature[i]
      if (clust %in% rownames(all_z_scores) && feat %in% colnames(all_z_scores)) {
        heatmap_df$z_score[i] <- all_z_scores[clust, feat]
      }
    }
    
    # Remove rows with missing data
    heatmap_df <- heatmap_df[!is.na(heatmap_df$z_score), ]
    
    if (nrow(heatmap_df) > 0) {
      p_combined <- ggplot(heatmap_df, aes(x = cluster, y = feature, fill = z_score)) +
        geom_tile() +
        scale_fill_gradient2(low = "#E74C3C", mid = "white", high = "#3498DB",
                            midpoint = 0, name = "Z-Score") +
        labs(title = "Feature Importance Across Hierarchical Clusters",
             subtitle = "Top 30 most distinguishing features (by variance across clusters)",
             x = "Cluster",
             y = "Feature") +
        theme_minimal() +
        theme(axis.text.y = element_text(size = 7),
              axis.text.x = element_text(size = 10, face = "bold"),
              plot.title = element_text(size = 14, face = "bold"),
              plot.subtitle = element_text(size = 10))
      
      ggsave(file.path(opt$`output-dir`, "hierarchical", "hierarchical_all_clusters_feature_heatmap.png"), 
             p_combined, width = 10, height = 14, dpi = 300)
    }
  }
  
  # 5. DBSCAN visualization
  if (file.exists(cluster_files$dbscan)) {
    cat("  [9/11] DBSCAN Clusters in PCA Space\n")
    clusters_db <- read.csv(cluster_files$dbscan, stringsAsFactors = FALSE)
    clusters_db <- clusters_db[match(processed_df$bout_id, clusters_db$bout_id), ]
    
    pca_df <- data.frame(
      PC1 = pca_result$x[, 1],
      PC2 = pca_result$x[, 2],
      cluster = as.factor(clusters_db$cluster)
    )
    
    p <- ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster)) +
      geom_point(alpha = 0.6, size = 1) +
      scale_color_manual(values = c("0" = "gray", "1" = "steelblue", "2" = "darkgreen", 
                                    "3" = "orange", "4" = "purple", "5" = "red")) +
      labs(title = "DBSCAN Clusters in PCA Space (0 = Noise)",
           x = "Principal Component 1",
           y = "Principal Component 2") +
      theme_minimal()
    
    ggsave(file.path(opt$`output-dir`, "dbscan", "dbscan_clusters_pca.png"), p, 
           width = 12, height = 10, dpi = 300)
    
    # DBSCAN cluster sizes
    cat("  [10/11] DBSCAN Cluster Sizes\n")
    cluster_sizes <- table(clusters_db$cluster)
    cluster_size_df <- data.frame(
      cluster = names(cluster_sizes),
      count = as.numeric(cluster_sizes)
    )
    
    p <- ggplot(cluster_size_df, aes(x = cluster, y = count, fill = cluster)) +
      geom_bar(stat = "identity") +
      scale_fill_manual(values = c("0" = "gray", "1" = "steelblue", "2" = "darkgreen", 
                                  "3" = "orange", "4" = "purple", "5" = "red")) +
      labs(title = "DBSCAN Cluster Sizes (0 = Noise)",
           x = "Cluster ID",
           y = "Number of Bouts") +
      theme_minimal() +
      theme(legend.position = "none")
    
    ggsave(file.path(opt$`output-dir`, "dbscan", "dbscan_cluster_sizes.png"), p, 
           width = 8, height = 6, dpi = 300)
  }
  
  # 6. Comparison of all methods
  cat("  [11/11] Comparison of All Clustering Methods\n")
  if (file.exists(cluster_files$kmeans) && file.exists(cluster_files$hierarchical) && 
      file.exists(cluster_files$dbscan)) {
    clusters_km <- read.csv(cluster_files$kmeans, stringsAsFactors = FALSE)
    clusters_hier <- read.csv(cluster_files$hierarchical, stringsAsFactors = FALSE)
    clusters_db <- read.csv(cluster_files$dbscan, stringsAsFactors = FALSE)
    
    pca_df <- data.frame(
      PC1 = pca_result$x[, 1],
      PC2 = pca_result$x[, 2],
      kmeans = as.factor(clusters_km$cluster[match(processed_df$bout_id, clusters_km$bout_id)]),
      hierarchical = as.factor(clusters_hier$cluster[match(processed_df$bout_id, clusters_hier$bout_id)]),
      dbscan = as.factor(clusters_db$cluster[match(processed_df$bout_id, clusters_db$bout_id)])
    )
    
    p1 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = kmeans)) +
      geom_point(alpha = 0.5, size = 0.8) +
      scale_color_brewer(palette = "Set2") +
      labs(title = "K-means", x = "PC1", y = "PC2") +
      theme_minimal() + theme(legend.position = "none")
    
    p2 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = hierarchical)) +
      geom_point(alpha = 0.5, size = 0.8) +
      scale_color_brewer(palette = "Set2") +
      labs(title = "Hierarchical", x = "PC1", y = "PC2") +
      theme_minimal() + theme(legend.position = "none")
    
    p3 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = dbscan)) +
      geom_point(alpha = 0.5, size = 0.8) +
      scale_color_manual(values = c("0" = "gray", "1" = "steelblue", "2" = "darkgreen")) +
      labs(title = "DBSCAN", x = "PC1", y = "PC2") +
      theme_minimal() + theme(legend.position = "none")
    
    combined <- grid.arrange(p1, p2, p3, ncol = 3, 
                            top = "Comparison of Clustering Methods in PCA Space")
    
    ggsave(file.path(opt$`output-dir`, "clustering_comparison.png"), combined, 
           width = 18, height = 6, dpi = 300)
  }
  
  cat("\n============================================================\n")
  cat("Clustering visualizations complete!\n")
  cat(sprintf("Plots saved to: %s\n", opt$`output-dir`))
  cat("\nGenerated plots:\n")
  cat("  Shared (main folder):\n")
  cat("    - pca_scree_plot.png\n")
  cat("    - pca_biplot.png\n")
  cat("    - clustering_comparison.png\n")
  cat("  K-means (kmeans/):\n")
  cat("    - kmeans_clusters_pca.png\n")
  cat("    - kmeans_cluster_sizes.png\n")
  cat("  Hierarchical (hierarchical/):\n")
  cat("    - hierarchical_dendrogram.png\n")
  cat("    - hierarchical_clusters_pca.png\n")
  cat("    - hierarchical_cluster_*_features.png (feature importance per cluster)\n")
  cat("    - hierarchical_cluster_*_heatmap.png (feature comparison per cluster)\n")
  cat("    - hierarchical_all_clusters_feature_heatmap.png (combined heatmap)\n")
  cat("  DBSCAN (dbscan/):\n")
  cat("    - dbscan_clusters_pca.png\n")
  cat("    - dbscan_cluster_sizes.png\n")
  
  # Ensure all graphics devices are closed
  while (dev.cur() != 1) {
    dev.off()
  }
  
  # Move Rplots.pdf to output directory if it was created
  if (file.exists("Rplots.pdf")) {
    file.rename("Rplots.pdf", file.path(opt$`output-dir`, "Rplots.pdf"))
    cat("Note: Rplots.pdf moved to output directory\n")
  }
}

# Run main
main()

# Final cleanup: close any remaining graphics devices
while (dev.cur() != 1) {
  dev.off()
}

# Move Rplots.pdf to output directory if it was created in main directory
if (file.exists("Rplots.pdf")) {
  # Try to move to clustering directory (default)
  output_dir <- "results/clustering"
  if (dir.exists(output_dir)) {
    file.rename("Rplots.pdf", file.path(output_dir, "Rplots.pdf"))
  }
}

