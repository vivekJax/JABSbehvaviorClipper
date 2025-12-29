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
  library(parallel)
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
              help="Enable verbose logging"),
  make_option(c("--ncores"), type="integer", default=NULL,
              help="Number of CPU cores to use for parallel processing (default: CPU cores - 1)")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Set up parallel processing (default: n-1 cores)
if (is.null(opt$ncores)) {
  # Get CPU count using Python (more reliable cross-platform)
  cpu_cores_str <- system("python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())'", intern=TRUE)
  cpu_cores <- as.integer(cpu_cores_str[1])
  n_cores <- max(1, cpu_cores - 1)
} else {
  n_cores <- max(1, opt$ncores)
}

# Set up parallel backend (will be created after functions are defined)
cl <- NULL
n_cores_global <- n_cores

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

# Helper function to evaluate k for parallel processing
evaluate_k <- function(k, X) {
  km <- kmeans(X, centers=k, nstart=25, iter.max=300)  # Use same parameters as main function
  sil <- silhouette(km$cluster, dist(X))
  return(mean(sil[, "sil_width"]))
}

# Find optimal number of clusters using multiple criteria (statistically robust)
find_optimal_k <- function(X, max_k = 10, cl = NULL) {
  n_samples <- nrow(X)
  if (n_samples < max_k) {
    max_k <- max(2, n_samples - 1)
  }
  
  k_vals <- 2:max_k  # Use k_vals instead of k_range to avoid conflicts
  
  # Calculate multiple metrics for robust k selection
  silhouette_scores <- numeric(length(k_vals))
  withinss_scores <- numeric(length(k_vals))
  
  # Use parallel processing if cluster is available
  if (!is.null(cl) && length(k_vals) > 1) {
    # Export X and helper function to cluster nodes
    # Note: k_vals is passed directly to parSapply, not exported
    clusterExport(cl, c("X", "evaluate_k"), envir=environment())
    silhouette_scores <- parSapply(cl, k_vals, function(k) evaluate_k(k, X))
    
    # Calculate within-cluster sum of squares (for elbow method)
    # This must be done sequentially as it's not parallelized
    for (i in seq_along(k_vals)) {
      k <- k_vals[i]
      km <- kmeans(X, centers=k, nstart=25, iter.max=300)  # Increased nstart for stability
      withinss_scores[i] <- km$tot.withinss
    }
  } else {
    for (i in seq_along(k_vals)) {
      k <- k_vals[i]
      silhouette_scores[i] <- evaluate_k(k, X)
      km <- kmeans(X, centers=k, nstart=25, iter.max=300)  # Increased nstart for stability
      withinss_scores[i] <- km$tot.withinss
    }
  }
  
  # Primary criterion: silhouette score (higher is better)
  best_sil_idx <- which.max(silhouette_scores)
  optimal_k_sil <- k_vals[best_sil_idx]
  
  # Secondary criterion: elbow method (look for largest drop in withinss)
  # Calculate percentage decrease in withinss
  if (length(withinss_scores) > 1) {
    withinss_decrease <- -diff(withinss_scores) / withinss_scores[-length(withinss_scores)]
    # Find elbow: largest decrease (but penalize very high k)
    elbow_scores <- withinss_decrease - 0.1 * (k_vals[-1] - 2)  # Penalty for higher k
    best_elbow_idx <- which.max(elbow_scores)
    optimal_k_elbow <- k_vals[best_elbow_idx + 1]
  } else {
    optimal_k_elbow <- optimal_k_sil
  }
  
  # Use silhouette as primary, but report both
  optimal_k <- optimal_k_sil
  
  cat(sprintf("Optimal k selection:\n"))
  cat(sprintf("  Silhouette method: k=%d (score: %.3f)\n", 
             optimal_k_sil, silhouette_scores[best_sil_idx]))
  if (length(withinss_scores) > 1) {
    cat(sprintf("  Elbow method: k=%d\n", optimal_k_elbow))
  }
  cat(sprintf("  Selected: k=%d\n", optimal_k))
  
  return(optimal_k)
}

# Perform K-means clustering
perform_kmeans_clustering <- function(X, n_clusters = NULL, cl = NULL) {
  if (is.null(n_clusters)) {
    n_clusters <- find_optimal_k(X, cl = cl)
  }
  
  # Increased nstart for better stability (statistical best practice: 25-50)
  km <- kmeans(X, centers=n_clusters, nstart=25, iter.max=300)
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
perform_hierarchical_clustering <- function(X, n_clusters = NULL, linkage = "ward.D2", cl = NULL) {
  if (is.null(n_clusters)) {
    n_clusters <- find_optimal_k(X, cl = cl)
  }
  
  # Statistical validation: Use appropriate distance metric
  # For continuous features, Euclidean is standard; Ward.D2 is preferred for non-Euclidean
  dist_matrix <- dist(X, method="euclidean")
  
  # Ward.D2 is preferred over Ward.D (more robust, handles non-Euclidean distances better)
  # It's appropriate when clusters are expected to be spherical/compact
  hc <- hclust(dist_matrix, method=linkage)
  
  # cutree assigns cluster IDs based on dendrogram order (arbitrary but consistent)
  # Cluster IDs are labels, not ordered by size - this is statistically correct
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

# Estimate optimal eps using k-distance graph method (statistical best practice)
# This finds the "knee" in the sorted k-nearest neighbor distances
estimate_eps <- function(X, minPts, k_percentile = 0.95) {
  n_samples <- nrow(X)
  if (n_samples < minPts + 1) {
    return(0.5)  # Fallback
  }
  
  # For large datasets, sample to speed up computation
  if (n_samples > 1000) {
    sample_idx <- sample(n_samples, min(1000, n_samples))
    X_sample <- X[sample_idx, , drop=FALSE]
  } else {
    X_sample <- X
  }
  
  # Calculate k-nearest neighbor distances for each point
  # Use dbscan::kNN for efficient computation
  k_distances <- numeric(nrow(X_sample))
  
  for (i in 1:nrow(X_sample)) {
    # Calculate distances from point i to all other points
    distances <- sqrt(rowSums((X_sample - matrix(X_sample[i, ], nrow=nrow(X_sample), 
                                                  ncol=ncol(X_sample), byrow=TRUE))^2))
    # Get kth nearest neighbor distance (excluding self)
    sorted_dist <- sort(distances[-i])
    if (length(sorted_dist) >= minPts) {
      k_distances[i] <- sorted_dist[minPts]
    } else {
      k_distances[i] <- sorted_dist[length(sorted_dist)]
    }
  }
  
  # Sort k-distances and find knee (sharp increase)
  sorted_k_dist <- sort(k_distances)
  
  # Use percentile method: eps is the k_percentile-th percentile of k-distances
  # This captures the point where distances start increasing sharply
  eps_estimate <- quantile(sorted_k_dist, probs=k_percentile, na.rm=TRUE)
  
  # Alternative: find knee using second derivative (more sophisticated)
  # Calculate first and second differences
  if (length(sorted_k_dist) > 10) {
    first_diff <- diff(sorted_k_dist)
    second_diff <- diff(first_diff)
    
    # Find point of maximum curvature (largest second derivative)
    if (any(!is.na(second_diff) & is.finite(second_diff))) {
      knee_idx <- which.max(second_diff)
      if (knee_idx > 0 && knee_idx <= length(sorted_k_dist)) {
        eps_knee <- sorted_k_dist[knee_idx]
        # Use the larger of percentile or knee (more conservative)
        eps_estimate <- max(eps_estimate, eps_knee, na.rm=TRUE)
      }
    }
  }
  
  return(as.numeric(eps_estimate))
}

# Perform DBSCAN clustering with automatic parameter estimation
# DBSCAN is density-based and doesn't require pre-specifying number of clusters
# eps: maximum distance between points in the same cluster
# minPts: minimum number of points to form a dense region
perform_dbscan_clustering <- function(X, eps = NULL, minPts = NULL) {
  if (!requireNamespace("dbscan", quietly=TRUE)) {
    stop("Package 'dbscan' is required for DBSCAN clustering. Install with: install.packages('dbscan')")
  }
  
  n_samples <- nrow(X)
  n_dims <- ncol(X)
  
  # Statistical best practice: minPts should be at least dimensions + 1
  # For high-dimensional data, use a more conservative approach
  # Rule of thumb: minPts = 2 * dimensions, but cap at reasonable values
  if (is.null(minPts)) {
    # For high-dimensional data, use a smaller minPts relative to sample size
    minPts <- max(4, min(floor(n_samples / 20), max(5, n_dims)))
    cat(sprintf("DBSCAN: Using adaptive minPts=%d (based on %d samples, %d dimensions)\n", 
                minPts, n_samples, n_dims))
  }
  
  # Automatically estimate eps if not provided
  if (is.null(eps)) {
    cat("DBSCAN: Automatically estimating eps using k-distance graph method...\n")
    eps_initial <- estimate_eps(X, minPts, k_percentile=0.85)  # Use 85th percentile
    cat(sprintf("DBSCAN: Initial eps estimate=%.3f\n", eps_initial))
    
    # Try multiple eps values to find optimal balance
    # For high-dimensional data, we need to explore a wider range
    # Start with smaller eps values to get more clusters
    eps_base <- max(0.5, min(eps_initial, 20))  # Cap at reasonable range
    
    # Generate a wider range of eps candidates, focusing on smaller values
    # to encourage more clusters
    eps_candidates <- c(
      eps_base * 0.1, eps_base * 0.2, eps_base * 0.3, eps_base * 0.4,
      eps_base * 0.5, eps_base * 0.6, eps_base * 0.7, eps_base * 0.85,
      eps_base, eps_base * 1.2, eps_base * 1.5
    )
    eps_candidates <- unique(sort(eps_candidates))
    
    cat("DBSCAN: Testing multiple eps values to find optimal clustering...\n")
    best_eps <- eps_candidates[1]
    best_n_clusters <- 0
    best_n_noise_ratio <- 1.0
    best_score <- -Inf
    
    # Also try some fixed smaller eps values for high-dimensional data
    # In high dimensions, distances are larger, so we need proportionally larger eps
    # But to get more clusters, we need smaller eps relative to the data scale
    fixed_eps_candidates <- c(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    all_eps_candidates <- unique(sort(c(eps_candidates, fixed_eps_candidates)))
    
    for (eps_candidate in all_eps_candidates) {
      db_test <- dbscan::dbscan(X, eps=eps_candidate, minPts=minPts)
      unique_clusters_test <- unique(db_test$cluster)
      n_clusters_test <- length(unique_clusters_test[unique_clusters_test != 0])
      n_noise_test <- sum(db_test$cluster == 0)
      noise_ratio <- n_noise_test / n_samples
      
      # Score: strongly prefer multiple clusters, allow moderate noise
      # Ideal: 2-10 clusters, <40% noise
      if (n_clusters_test >= 1 && n_clusters_test <= 15) {
        # Score heavily favors multiple clusters
        # Bonus for 2-8 clusters, penalty for too many (>10) or too much noise (>50%)
        cluster_bonus <- if (n_clusters_test >= 2 && n_clusters_test <= 8) {
          n_clusters_test * 2  # Strong bonus for reasonable number of clusters
        } else if (n_clusters_test == 1) {
          0.5  # Low score for single cluster
        } else {
          n_clusters_test - 5  # Penalty for too many clusters
        }
        
        noise_penalty <- if (noise_ratio > 0.5) {
          noise_ratio * 20  # Heavy penalty for >50% noise
        } else if (noise_ratio > 0.3) {
          noise_ratio * 5   # Moderate penalty for 30-50% noise
        } else {
          noise_ratio * 2   # Light penalty for <30% noise
        }
        
        score <- cluster_bonus - noise_penalty
        
        if (score > best_score || (score == best_score && n_clusters_test > best_n_clusters)) {
          best_eps <- eps_candidate
          best_n_clusters <- n_clusters_test
          best_n_noise_ratio <- noise_ratio
          best_score <- score
        }
      }
    }
    
    eps <- best_eps
    cat(sprintf("DBSCAN: Selected eps=%.3f (found %d clusters, %.1f%% noise, score=%.2f)\n", 
                eps, best_n_clusters, best_n_noise_ratio * 100, best_score))
    
    # If still only 1 cluster, try reducing minPts more aggressively
    if (best_n_clusters <= 1 && minPts > 4) {
      cat("DBSCAN: Only 1 cluster found, trying with reduced minPts and smaller eps...\n")
      # Try multiple reduced minPts values, starting from smallest
      minPts_options <- c(max(4, minPts - 4), max(4, minPts - 3), max(4, minPts - 2), max(4, minPts - 1))
      minPts_options <- unique(sort(minPts_options, decreasing=FALSE))
      
      # Focus on smaller eps values to encourage more clusters
      # Based on testing, need eps around 5-8 for this data
      small_eps_candidates <- c(4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0)
      
      # Reset best to allow finding solutions with >1 cluster
      best_n_clusters_loop <- best_n_clusters
      best_eps_loop <- best_eps
      best_n_noise_ratio_loop <- best_n_noise_ratio
      
      for (minPts_reduced in minPts_options) {
        cat(sprintf("  Testing minPts=%d with smaller eps values...\n", minPts_reduced))
        for (eps_candidate in small_eps_candidates) {
          db_test <- dbscan::dbscan(X, eps=eps_candidate, minPts=minPts_reduced)
          unique_clusters_test <- unique(db_test$cluster)
          n_clusters_test <- length(unique_clusters_test[unique_clusters_test != 0])
          n_noise_test <- sum(db_test$cluster == 0)
          noise_ratio <- n_noise_test / n_samples
          
          # Accept if we get more clusters (even with high noise)
          # For high-dimensional dense data, we may need to accept up to 90% noise to get multiple clusters
          if (n_clusters_test > best_n_clusters_loop && noise_ratio < 0.95) {
            # Prefer solutions with 2-10 clusters and <75% noise
            best_eps_loop <- eps_candidate
            best_n_clusters_loop <- n_clusters_test
            best_n_noise_ratio_loop <- noise_ratio
            minPts <- minPts_reduced
            cat(sprintf("  ✓ Better: minPts=%d, eps=%.3f → %d clusters, %.1f%% noise\n",
                       minPts, eps_candidate, n_clusters_test, noise_ratio * 100))
            # If we found a good solution (2+ clusters with reasonable noise), we can stop
            if (n_clusters_test >= 2 && noise_ratio < 0.5) {
              break
            }
          } else if (n_clusters_test == best_n_clusters_loop && n_clusters_test > 1 && 
                     noise_ratio < best_n_noise_ratio_loop && noise_ratio < 0.95) {
            # Same number of clusters but less noise
            best_eps_loop <- eps_candidate
            best_n_noise_ratio_loop <- noise_ratio
            minPts <- minPts_reduced
            cat(sprintf("  ✓ Better noise: minPts=%d, eps=%.3f → %d clusters, %.1f%% noise\n",
                       minPts, eps_candidate, n_clusters_test, noise_ratio * 100))
          }
        }
        # If we found a good solution, stop trying other minPts
        if (best_n_clusters_loop >= 2 && best_n_noise_ratio_loop < 0.5) {
          break
        }
      }
      
      # Update best values if we found something better
      if (best_n_clusters_loop > best_n_clusters || 
          (best_n_clusters_loop == best_n_clusters && best_n_clusters_loop > 1 && 
           best_n_noise_ratio_loop < best_n_noise_ratio)) {
        best_eps <- best_eps_loop
        best_n_clusters <- best_n_clusters_loop
        best_n_noise_ratio <- best_n_noise_ratio_loop
        eps <- best_eps
        cat(sprintf("DBSCAN: Updated to %d clusters with eps=%.3f, minPts=%d (%.1f%% noise)\n",
                   best_n_clusters, eps, minPts, best_n_noise_ratio * 100))
      } else {
        # If we didn't find better, but original was only 1 cluster, 
        # accept the best multi-cluster solution even with high noise
        if (best_n_clusters <= 1 && best_n_clusters_loop > 1) {
          best_eps <- best_eps_loop
          best_n_clusters <- best_n_clusters_loop
          best_n_noise_ratio <- best_n_noise_ratio_loop
          eps <- best_eps
          cat(sprintf("DBSCAN: Accepting %d clusters with eps=%.3f, minPts=%d (%.1f%% noise) - better than 1 cluster\n",
                     best_n_clusters, eps, minPts, best_n_noise_ratio * 100))
        }
      }
    }
  }
  
  # DBSCAN automatically determines number of clusters based on density
  # Cluster 0 represents noise/outliers (statistically important)
  db <- dbscan::dbscan(X, eps=eps, minPts=minPts)
  labels <- db$cluster
  
  # Count clusters (excluding noise cluster 0)
  unique_clusters <- unique(labels)
  n_clusters <- length(unique_clusters[unique_clusters != 0])
  n_noise <- sum(labels == 0)
  noise_ratio <- n_noise / n_samples
  
  info <- list(
    method = "dbscan",
    n_clusters = n_clusters,
    n_noise = n_noise,
    noise_ratio = noise_ratio,
    eps = eps,
    minPts = minPts
  )
  
  if (n_clusters > 1) {
    dist_matrix <- dist(X)
    sil <- silhouette(labels, dist_matrix)
    silhouette_score <- mean(sil[, "sil_width"])
    ch_score <- tryCatch({
      if (requireNamespace("fpc", quietly=TRUE)) {
        fpc::calinhara(X, labels)
      } else {
        NA
      }
    }, error = function(e) NA)
    
    info$silhouette_score <- silhouette_score
    info$calinski_harabasz_score <- ch_score
    
    cat(sprintf("DBSCAN clustering: %d clusters, %d noise points (%.1f%%), silhouette=%.3f\n", 
               n_clusters, n_noise, noise_ratio * 100, silhouette_score))
  } else {
    cat(sprintf("DBSCAN found %d clusters (%.1f%% noise). Parameters: eps=%.3f, minPts=%d\n", 
               n_clusters, noise_ratio * 100, eps, minPts))
    if (noise_ratio > 0.8) {
      cat("  Warning: >80%% noise. Consider increasing eps or decreasing minPts.\n")
    }
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
main <- function(cl = NULL) {
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
        result <- perform_kmeans_clustering(X_scaled, n_clusters=opt$`n-clusters`, cl=cl)
      } else if (method == "hierarchical") {
        result <- perform_hierarchical_clustering(X_scaled, n_clusters=opt$`n-clusters`, cl=cl)
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

# Set up parallel backend after functions are defined
if (n_cores_global > 1) {
  cl <- makeCluster(n_cores_global)
  clusterExport(cl, c("evaluate_k", "find_optimal_k", "kmeans", "silhouette", "dist"))
  cat(sprintf("Using %d CPU cores for parallel processing\n", n_cores_global))
} else {
  cl <- NULL
  cat("Using sequential processing (1 core)\n")
}

# Run main function
if (!interactive()) {
  tryCatch({
    main(cl = cl)
  }, finally = {
    # Clean up parallel cluster
    if (!is.null(cl)) {
      stopCluster(cl)
    }
  })
}

