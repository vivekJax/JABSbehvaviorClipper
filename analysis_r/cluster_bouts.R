#!/usr/bin/env Rscript
# Cluster behavior bouts based on extracted features.
#
# This script:
# 1. Loads extracted bout features
# 2. Preprocesses and scales features
# 3. Performs clustering using multiple algorithms
# 4. Evaluates cluster quality
# 5. Saves cluster assignments and statistics

# Add default library paths (user library first)
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(cluster)
  library(factoextra)
  library(NbClust)
  library(jsonlite)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-i", "--input"), type="character", default="bout_features.csv",
              help="Input CSV file with bout features (default: bout_features.csv)"),
  make_option(c("-m", "--method"), type="character", default="kmeans",
              help="Clustering method: kmeans, hierarchical, dbscan, gmm, or all (default: kmeans)"),
  make_option(c("-k", "--n-clusters"), type="integer", default=NULL,
              help="Number of clusters (auto-detect if not specified)"),
  make_option(c("-s", "--scale-method"), type="character", default="standard",
              help="Feature scaling method: standard, minmax, or robust (default: standard)"),
  make_option(c("-o", "--output-dir"), type="character", default=".",
              help="Output directory for results (default: current directory)"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Separate metadata and features
separate_metadata_and_features <- function(df) {
  metadata_cols <- c("bout_id", "video_name", "animal_id", "start_frame", 
                     "end_frame", "behavior", "duration_frames")
  
  existing_metadata_cols <- intersect(metadata_cols, colnames(df))
  feature_cols <- setdiff(colnames(df), existing_metadata_cols)
  
  metadata_df <- df[, existing_metadata_cols, drop=FALSE]
  features_df <- df[, feature_cols, drop=FALSE]
  
  return(list(metadata=metadata_df, features=features_df))
}

# Handle missing values
handle_missing_values <- function(df, strategy = "mean") {
  if (strategy == "drop") {
    # Drop columns with all NA
    df <- df[, colSums(!is.na(df)) > 0, drop=FALSE]
    # Drop rows with any NA
    df <- df[complete.cases(df), , drop=FALSE]
    cat(sprintf("Dropped rows/columns with missing values. Remaining shape: %d, %d\n", 
               nrow(df), ncol(df)))
  } else if (strategy == "mean") {
    # Fill missing values with column means
    for (col in colnames(df)) {
      if (any(is.na(df[[col]]))) {
        col_mean <- mean(df[[col]], na.rm=TRUE)
        if (!is.na(col_mean)) {
          df[[col]][is.na(df[[col]])] <- col_mean
        } else {
          # If mean is NA (all values NA), fill with 0
          df[[col]][is.na(df[[col]])] <- 0
        }
      }
    }
    cat("Filled missing values with mean\n")
  } else if (strategy == "median") {
    # Fill missing values with column medians
    for (col in colnames(df)) {
      if (any(is.na(df[[col]]))) {
        col_median <- median(df[[col]], na.rm=TRUE)
        if (!is.na(col_median)) {
          df[[col]][is.na(df[[col]])] <- col_median
        } else {
          df[[col]][is.na(df[[col]])] <- 0
        }
      }
    }
    cat("Filled missing values with median\n")
  }
  
  return(df)
}

# Scale features
scale_features <- function(X, method = "standard") {
  if (method == "standard") {
    X_scaled <- scale(X)
  } else if (method == "minmax") {
    min_vals <- apply(X, 2, min, na.rm=TRUE)
    max_vals <- apply(X, 2, max, na.rm=TRUE)
    X_scaled <- sweep(sweep(X, 2, min_vals, "-"), 2, max_vals - min_vals, "/")
  } else if (method == "robust") {
    median_vals <- apply(X, 2, median, na.rm=TRUE)
    mad_vals <- apply(X, 2, mad, na.rm=TRUE)
    X_scaled <- sweep(sweep(X, 2, median_vals, "-"), 2, mad_vals, "/")
  }
  
  cat(sprintf("Scaled features using %s scaling\n", method))
  return(X_scaled)
}

# Find optimal number of clusters
find_optimal_k <- function(X, max_k = 10) {
  n_samples <- nrow(X)
  if (n_samples < max_k) {
    max_k <- max(2, n_samples - 1)
  }
  
  k_range <- 2:max_k
  silhouette_scores <- numeric(length(k_range))
  
  for (i in seq_along(k_range)) {
    k <- k_range[i]
    km <- kmeans(X, centers=k, nstart=10)
    sil <- silhouette(km$cluster, dist(X))
    silhouette_scores[i] <- mean(sil[, "sil_width"])
  }
  
  best_k_idx <- which.max(silhouette_scores)
  optimal_k <- k_range[best_k_idx]
  
  cat(sprintf("Optimal k based on silhouette score: %d (score: %.3f)\n", 
             optimal_k, silhouette_scores[best_k_idx]))
  
  return(optimal_k)
}

# Perform K-means clustering
perform_kmeans_clustering <- function(X, n_clusters = NULL) {
  if (is.null(n_clusters)) {
    n_clusters <- find_optimal_k(X)
  }
  
  km <- kmeans(X, centers=n_clusters, nstart=10)
  labels <- km$cluster
  
  # Calculate metrics
  sil <- silhouette(km$cluster, dist(X))
  silhouette_score <- mean(sil[, "sil_width"])
  
  # Calinski-Harabasz index (using factoextra if available, otherwise skip)
  ch_score <- tryCatch({
    if (requireNamespace("fpc", quietly=TRUE)) {
      fpc::calinhara(X, km$cluster)
    } else {
      NA
    }
  }, error = function(e) NA)
  
  info <- list(
    method = "kmeans",
    n_clusters = n_clusters,
    inertia = km$tot.withinss,
    silhouette_score = silhouette_score,
    calinski_harabasz_score = ch_score,
    centroids = km$centers
  )
  
  cat(sprintf("K-means clustering: %d clusters, silhouette=%.3f\n", 
             n_clusters, silhouette_score))
  
  return(list(labels=labels, info=info))
}

# Perform hierarchical clustering
perform_hierarchical_clustering <- function(X, n_clusters = NULL, linkage = "ward.D2") {
  if (is.null(n_clusters)) {
    n_clusters <- find_optimal_k(X)
  }
  
  dist_matrix <- dist(X)
  hc <- hclust(dist_matrix, method=linkage)
  labels <- cutree(hc, k=n_clusters)
  
  # Calculate metrics
  sil <- silhouette(labels, dist_matrix)
  silhouette_score <- mean(sil[, "sil_width"])
  ch_score <- tryCatch({
    if (requireNamespace("fpc", quietly=TRUE)) {
      fpc::calinhara(X, labels)
    } else {
      NA
    }
  }, error = function(e) NA)
  
  info <- list(
    method = "hierarchical",
    n_clusters = n_clusters,
    linkage = linkage,
    silhouette_score = silhouette_score,
    calinski_harabasz_score = ch_score
  )
  
  cat(sprintf("Hierarchical clustering: %d clusters, silhouette=%.3f\n", 
             n_clusters, silhouette_score))
  
  return(list(labels=labels, info=info))
}

# Perform DBSCAN clustering
perform_dbscan_clustering <- function(X, eps = 0.5, minPts = 5) {
  if (!requireNamespace("dbscan", quietly=TRUE)) {
    stop("Package 'dbscan' is required for DBSCAN clustering. Install with: install.packages('dbscan')")
  }
  
  db <- dbscan::dbscan(X, eps=eps, minPts=minPts)
  labels <- db$cluster
  
  n_clusters <- length(unique(labels)) - sum(labels == 0)
  n_noise <- sum(labels == 0)
  
  info <- list(
    method = "dbscan",
    n_clusters = n_clusters,
    n_noise = n_noise,
    eps = eps,
    minPts = minPts
  )
  
  if (n_clusters > 1) {
    dist_matrix <- dist(X)
    sil <- silhouette(labels, dist_matrix)
    silhouette_score <- mean(sil[, "sil_width"])
    ch_score <- calinhara(X, labels)
    
    info$silhouette_score <- silhouette_score
    info$calinski_harabasz_score <- ch_score
    
    cat(sprintf("DBSCAN clustering: %d clusters, %d noise points, silhouette=%.3f\n", 
               n_clusters, n_noise, silhouette_score))
  } else {
    cat(sprintf("DBSCAN found %d clusters (mostly noise). Try adjusting eps or minPts.\n", n_clusters))
  }
  
  return(list(labels=labels, info=info))
}

# Calculate cluster statistics
calculate_cluster_statistics <- function(X, labels, metadata_df) {
  stats <- list()
  unique_labels <- sort(unique(labels))
  
  for (cluster_id in unique_labels) {
    if (cluster_id == 0) next  # Skip noise points in DBSCAN
    
    cluster_mask <- labels == cluster_id
    cluster_data <- X[cluster_mask, , drop=FALSE]
    cluster_metadata <- metadata_df[cluster_mask, , drop=FALSE]
    
    stats[[as.character(cluster_id)]] <- list(
      size = sum(cluster_mask),
      mean_features = as.list(colMeans(cluster_data)),
      std_features = as.list(apply(cluster_data, 2, sd)),
      videos = unique(cluster_metadata$video_name),
      animals = unique(cluster_metadata$animal_id),
      n_videos = length(unique(cluster_metadata$video_name)),
      n_animals = length(unique(cluster_metadata$animal_id)),
      mean_duration = if ("duration_frames" %in% colnames(cluster_metadata)) {
        mean(cluster_metadata$duration_frames)
      } else NULL
    )
  }
  
  return(stats)
}

# Main execution
main <- function() {
  cat(sprintf("Loading features from %s\n", opt$input))
  df <- read.csv(opt$input, stringsAsFactors=FALSE)
  
  if (nrow(df) == 0) {
    stop("Input file is empty")
  }
  
  # Separate metadata and features
  separated <- separate_metadata_and_features(df)
  metadata_df <- separated$metadata
  features_df <- separated$features
  
  # Handle missing values (use mean imputation to preserve all rows)
  features_df <- handle_missing_values(features_df, strategy="mean")
  
  # Update metadata to match (remove rows that were dropped)
  metadata_df <- metadata_df[rownames(features_df), , drop=FALSE]
  
  # Convert to matrix and scale
  X <- as.matrix(features_df)
  
  # Replace infinite values with 0
  X[is.infinite(X)] <- 0
  X[is.nan(X)] <- 0
  
  X_scaled <- scale_features(X, method=opt$`scale-method`)
  
  # Check for any remaining invalid values
  if (any(!is.finite(X_scaled))) {
    X_scaled[!is.finite(X_scaled)] <- 0
    cat("Replaced remaining non-finite values with 0\n")
  }
  
  cat(sprintf("Prepared %d samples with %d features\n", nrow(X_scaled), ncol(X_scaled)))
  
  # Perform clustering
  methods_to_run <- if (opt$method == "all") {
    c("kmeans", "hierarchical", "dbscan")
  } else {
    opt$method
  }
  
  all_results <- list()
  
  for (method in methods_to_run) {
    cat(sprintf("\nPerforming %s clustering...\n", method))
    
    tryCatch({
      if (method == "kmeans") {
        result <- perform_kmeans_clustering(X_scaled, n_clusters=opt$`n-clusters`)
      } else if (method == "hierarchical") {
        result <- perform_hierarchical_clustering(X_scaled, n_clusters=opt$`n-clusters`)
      } else if (method == "dbscan") {
        result <- perform_dbscan_clustering(X_scaled)
      } else {
        next
      }
      
      # Calculate cluster statistics
      cluster_stats <- calculate_cluster_statistics(X_scaled, result$labels, metadata_df)
      result$info$cluster_statistics <- cluster_stats
      
      # Save results
      results_df <- cbind(metadata_df, 
                         cluster_id = result$labels,
                         cluster_method = method)
      
      output_file <- file.path(opt$`output-dir`, sprintf("cluster_assignments_%s.csv", method))
      write.csv(results_df, output_file, row.names=FALSE)
      cat(sprintf("Saved cluster assignments to %s\n", output_file))
      
      # Save cluster statistics
      stats_file <- file.path(opt$`output-dir`, sprintf("cluster_statistics_%s.json", method))
      write_json(result$info, stats_file, pretty=TRUE)
      cat(sprintf("Saved cluster statistics to %s\n", stats_file))
      
      all_results[[method]] <- result
      
    }, error = function(e) {
      cat(sprintf("Error in %s clustering: %s\n", method, e$message))
    })
  }
  
  # Print summary
  cat("\n============================================================\n")
  cat("Clustering Summary\n")
  cat("============================================================\n")
  
  for (method in names(all_results)) {
    info <- all_results[[method]]$info
    cat(sprintf("\n%s:\n", toupper(method)))
    cat(sprintf("  Clusters: %d\n", info$n_clusters))
    if (!is.null(info$silhouette_score)) {
      cat(sprintf("  Silhouette Score: %.3f\n", info$silhouette_score))
    }
    if (!is.null(info$calinski_harabasz_score)) {
      cat(sprintf("  Calinski-Harabasz Score: %.2f\n", info$calinski_harabasz_score))
    }
  }
}

# Run main function
if (!interactive()) {
  main()
}

