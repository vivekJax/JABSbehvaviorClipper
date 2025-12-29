#!/usr/bin/env Rscript
# Cluster behavior bouts to identify sub-behaviors
#
# This script:
# 1. Loads feature data
# 2. Performs data preprocessing
# 3. Applies PCA for dimensionality reduction
# 4. Performs clustering using K-means, hierarchical, and DBSCAN methods
# 5. Saves cluster assignments
#
# Usage:
#   Rscript analysis_r/cluster_bouts.R --input bout_features.csv --output-dir results/

# Add default library paths
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(factoextra)
  library(cluster)
  library(dbscan)
  library(parallel)
})

# Source utility functions
source("analysis_r/utils/data_preprocessing.R")

# Parse command line arguments
option_list <- list(
  make_option(c("-i", "--input"), type="character", default="results/bout_features_filtered.csv",
              help="Input CSV file with bout features (default: results/bout_features_filtered.csv)"),
  make_option(c("-o", "--output-dir"), type="character", default="results",
              help="Output directory for cluster assignments (default: results)"),
  make_option(c("-m", "--method"), type="character", default="all",
              help="Clustering method: kmeans, hierarchical, dbscan, or all (default: all)"),
  make_option(c("-k", "--n-clusters"), type="integer", default=NULL,
              help="Number of clusters for K-means (default: auto-detect using silhouette)"),
  make_option(c("--pca-variance"), type="numeric", default=0.95,
              help="Proportion of variance to retain in PCA (default: 0.95)"),
  make_option(c("--workers"), type="integer", default=NULL,
              help="Number of parallel workers (default: detect_cores() - 1)"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Perform PCA dimensionality reduction
perform_pca <- function(df_features, variance_threshold = 0.95) {
  cat("Performing PCA dimensionality reduction...\n")
  
  # Remove any remaining constant columns
  df_features <- remove_constant_features(df_features, numeric_only = TRUE)
  
  # Check for sufficient data
  if (ncol(df_features) == 0) {
    stop("No features available after removing constant columns")
  }
  
  if (nrow(df_features) < 2) {
    stop("Insufficient data for PCA (need at least 2 samples)")
  }
  
  # Perform PCA
  pca_result <- prcomp(df_features, center = TRUE, scale. = TRUE)
  
  # Calculate cumulative variance
  cumvar <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
  
  # Find number of components to retain
  n_components <- which(cumvar >= variance_threshold)[1]
  if (is.na(n_components)) {
    n_components <- min(length(cumvar), ncol(df_features))
  }
  
  cat(sprintf("Retaining %d PCA components (%.1f%% variance explained)\n", 
             n_components, cumvar[n_components] * 100))
  
  # Extract reduced features
  reduced_features <- pca_result$x[, 1:n_components, drop = FALSE]
  
  return(list(
    reduced_features = reduced_features,
    pca_result = pca_result,
    n_components = n_components,
    variance_explained = cumvar[n_components]
  ))
}

# K-means clustering
cluster_kmeans <- function(features, k = NULL, n_workers = NULL) {
  cat("\n[K-means Clustering]\n")
  
  # Get number of workers (default: detectCores() - 1)
  if (is.null(n_workers)) {
    n_workers <- max(1, detectCores() - 1)
  }
  
  if (is.null(k)) {
    cat("Finding optimal k using silhouette score...\n")
    # Try k from 2 to min(10, n_samples/2)
    max_k <- min(10, floor(nrow(features) / 2))
    if (max_k < 2) {
      k <- 2
    } else {
      # Parallelize k-means trials for different k values
      k_candidates <- 2:max_k
      
      if (length(k_candidates) > 1 && n_workers > 1) {
        cat(sprintf("Testing %d k values in parallel using %d workers...\n", length(k_candidates), n_workers))
        cl <- makeCluster(n_workers)
        
        # Function to test one k value
        test_k <- function(k_val) {
          tryCatch({
            km <- kmeans(features, centers = k_val, nstart = 10, iter.max = 100)
            sil <- silhouette(km$cluster, dist(features))
            return(mean(sil[, 3]))
          }, error = function(e) {
            return(-1)
          })
        }
        
        # Export features to cluster workers
        clusterExport(cl, "features", envir = environment())
        silhouette_scores <- parSapply(cl, k_candidates, test_k)
        stopCluster(cl)
      } else {
        # Sequential processing
        silhouette_scores <- numeric(max_k - 1)
        for (k_candidate in k_candidates) {
          tryCatch({
            km <- kmeans(features, centers = k_candidate, nstart = 10, iter.max = 100)
            sil <- silhouette(km$cluster, dist(features))
            silhouette_scores[k_candidate - 1] <- mean(sil[, 3])
          }, error = function(e) {
            silhouette_scores[k_candidate - 1] <<- -1
          })
        }
      }
      
      k <- which.max(silhouette_scores) + 1
      cat(sprintf("Optimal k = %d (silhouette score: %.3f)\n", k, max(silhouette_scores)))
    }
  }
  
  cat(sprintf("Running K-means with k = %d...\n", k))
  # K-means already uses multiple starts internally, but we can increase nstart
  km_result <- kmeans(features, centers = k, nstart = max(25, n_workers * 5), iter.max = 100)
  
  return(list(
    clusters = km_result$cluster,
    centers = km_result$centers,
    withinss = km_result$withinss,
    totss = km_result$totss,
    betweenss = km_result$betweenss
  ))
}

# Hierarchical clustering
cluster_hierarchical <- function(features, k = NULL) {
  cat("\n[Hierarchical Clustering]\n")
  
  cat("Computing distance matrix...\n")
  dist_matrix <- dist(features, method = "euclidean")
  
  cat("Building dendrogram (Ward's method)...\n")
  hc_result <- hclust(dist_matrix, method = "ward.D2")
  
  if (is.null(k)) {
    # Use dynamic tree cutting or default to 5
    k <- 5
    cat(sprintf("Using k = %d clusters\n", k))
  }
  
  cat(sprintf("Cutting tree at k = %d...\n", k))
  clusters <- cutree(hc_result, k = k)
  
  return(list(
    clusters = clusters,
    dendrogram = hc_result
  ))
}

# DBSCAN clustering
cluster_dbscan <- function(features, eps = NULL, minPts = 5) {
  cat("\n[DBSCAN Clustering]\n")
  
  if (is.null(eps)) {
    cat("Estimating eps using k-distance graph...\n")
    # Use kth nearest neighbor distance
    k_dist <- kNNdist(features, k = minPts)
    k_dist_sorted <- sort(k_dist)
    
    # Use elbow method: find point where distance increases sharply
    # Use median of k-distances as initial estimate
    eps <- median(k_dist_sorted)
    cat(sprintf("Estimated eps = %.3f\n", eps))
  }
  
  cat(sprintf("Running DBSCAN (eps = %.3f, minPts = %d)...\n", eps, minPts))
  dbscan_result <- dbscan(features, eps = eps, minPts = minPts)
  
  n_clusters <- length(unique(dbscan_result$cluster[dbscan_result$cluster != 0]))
  n_noise <- sum(dbscan_result$cluster == 0)
  
  cat(sprintf("Found %d clusters and %d noise points\n", n_clusters, n_noise))
  
  return(list(
    clusters = dbscan_result$cluster,
    eps = eps,
    minPts = minPts
  ))
}

# Main execution
main <- function() {
  cat("============================================================\n")
  cat("Behavior Bout Clustering Analysis\n")
  cat("============================================================\n\n")
  
  # Load features
  cat(sprintf("Loading features from: %s\n", opt$input))
  if (!file.exists(opt$input)) {
    stop(sprintf("Input file not found: %s", opt$input))
  }
  
  df <- read.csv(opt$input, stringsAsFactors = FALSE)
  cat(sprintf("Loaded %d bouts with %d columns\n", nrow(df), ncol(df)))
  
  # Prepare features
  cat("\nPreprocessing features...\n")
  exclude_cols <- c("bout_id", "video_name", "animal_id", "start_frame", "end_frame", "behavior")
  prep_result <- prepare_features(df, exclude_cols = exclude_cols)
  
  processed_df <- prep_result$processed_df
  feature_cols <- prep_result$feature_cols
  
  cat(sprintf("After preprocessing: %d features\n", length(feature_cols)))
  
  # Extract feature matrix
  feature_matrix <- as.matrix(processed_df[, feature_cols, drop = FALSE])
  
  # Remove any remaining non-finite values
  feature_matrix[!is.finite(feature_matrix)] <- 0
  
  # Perform PCA
  pca_result <- perform_pca(feature_matrix, variance_threshold = opt$`pca-variance`)
  reduced_features <- pca_result$reduced_features
  
  # Create output directory and clustering subdirectory
  dir.create(opt$`output-dir`, showWarnings = FALSE, recursive = TRUE)
  clustering_dir <- file.path(opt$`output-dir`, "clustering")
  dir.create(clustering_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Save PCA results (shared across all methods)
  pca_file <- file.path(clustering_dir, "pca_results.RData")
  save(pca_result, file = pca_file)
  cat(sprintf("\nPCA results saved to: %s\n", pca_file))
  
  # Perform clustering
  cluster_results <- list()
  
  # Determine number of workers for parallel processing
  n_workers <- if (is.null(opt$workers)) {
    max(1, detectCores() - 1)
  } else {
    opt$workers
  }
  cat(sprintf("\nUsing %d parallel workers for clustering\n", n_workers))
  
  if (opt$method %in% c("all", "kmeans")) {
    km_result <- cluster_kmeans(reduced_features, k = opt$`n-clusters`, n_workers = n_workers)
    cluster_results$kmeans <- km_result
    
    # Create method-specific directory
    kmeans_dir <- file.path(clustering_dir, "kmeans")
    dir.create(kmeans_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Save K-means assignments
    assignments <- data.frame(
      bout_id = processed_df$bout_id,
      video_name = processed_df$video_name,
      animal_id = processed_df$animal_id,
      start_frame = processed_df$start_frame,
      end_frame = processed_df$end_frame,
      cluster = km_result$clusters
    )
    
    output_file <- file.path(kmeans_dir, "cluster_assignments_kmeans.csv")
    write.csv(assignments, output_file, row.names = FALSE)
    cat(sprintf("K-means assignments saved to: %s\n", output_file))
  }
  
  if (opt$method %in% c("all", "hierarchical")) {
    hier_result <- cluster_hierarchical(reduced_features, k = opt$`n-clusters`)
    cluster_results$hierarchical <- hier_result
    
    # Create method-specific directory
    hierarchical_dir <- file.path(clustering_dir, "hierarchical")
    dir.create(hierarchical_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Save hierarchical assignments
    assignments <- data.frame(
      bout_id = processed_df$bout_id,
      video_name = processed_df$video_name,
      animal_id = processed_df$animal_id,
      start_frame = processed_df$start_frame,
      end_frame = processed_df$end_frame,
      cluster = hier_result$clusters
    )
    
    output_file <- file.path(hierarchical_dir, "cluster_assignments_hierarchical.csv")
    write.csv(assignments, output_file, row.names = FALSE)
    cat(sprintf("Hierarchical assignments saved to: %s\n", output_file))
  }
  
  if (opt$method %in% c("all", "dbscan")) {
    dbscan_result <- cluster_dbscan(reduced_features)
    cluster_results$dbscan <- dbscan_result
    
    # Create method-specific directory
    dbscan_dir <- file.path(clustering_dir, "dbscan")
    dir.create(dbscan_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Save DBSCAN assignments
    assignments <- data.frame(
      bout_id = processed_df$bout_id,
      video_name = processed_df$video_name,
      animal_id = processed_df$animal_id,
      start_frame = processed_df$start_frame,
      end_frame = processed_df$end_frame,
      cluster = dbscan_result$clusters
    )
    
    output_file <- file.path(dbscan_dir, "cluster_assignments_dbscan.csv")
    write.csv(assignments, output_file, row.names = FALSE)
    cat(sprintf("DBSCAN assignments saved to: %s\n", output_file))
  }
  
  cat("\n============================================================\n")
  cat("Clustering complete!\n")
}

# Run main
main()

