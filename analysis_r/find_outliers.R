#!/usr/bin/env Rscript
# Find outlier behavior bouts using multiple distance-based methods
#
# This script:
# 1. Loads feature data and cluster assignments
# 2. Calculates outlier scores using Mahalanobis distance, LOF, and Isolation Forest
# 3. Identifies top outliers
# 4. Generates comprehensive explanation reports
#
# Usage:
#   Rscript analysis_r/find_outliers.R --features bout_features.csv --output-dir results/

# Add default library paths
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(MASS)
  library(dbscan)
  library(isotree)
  library(jsonlite)
  library(parallel)
})

# Source utility functions
source("analysis_r/utils/data_preprocessing.R")

# Parse command line arguments
option_list <- list(
  make_option(c("-f", "--features"), type="character", default="results/bout_features.csv",
              help="Input CSV file with bout features (default: results/bout_features.csv)"),
  make_option(c("-c", "--clusters"), type="character", default=NULL,
              help="Cluster assignments CSV file (optional)"),
  make_option(c("-o", "--output-dir"), type="character", default="results",
              help="Output directory for outlier results (default: results)"),
  make_option(c("--distance-metric"), type="character", default="mahalanobis",
              help="Primary distance metric: mahalanobis, lof, isolation (default: mahalanobis)"),
  make_option(c("--use-pca"), action="store_true", default=FALSE,
              help="Use PCA-reduced features for distance calculation"),
  make_option(c("--pca-variance"), type="numeric", default=0.95,
              help="Proportion of variance to retain in PCA (default: 0.95)"),
  make_option(c("--top-n"), type="integer", default=NULL,
              help="Number of top outliers to select (default: top 5%%)"),
  make_option(c("--percentile"), type="numeric", default=0.95,
              help="Percentile threshold for outliers (default: 0.95)"),
  make_option(c("--workers"), type="integer", default=NULL,
              help="Number of parallel workers (default: detect_cores() - 1)"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Calculate Mahalanobis distance
calculate_mahalanobis <- function(features) {
  cat("Calculating Mahalanobis distances...\n")
  
  # Calculate mean and covariance
  mu <- colMeans(features, na.rm = TRUE)
  
  # Handle singular covariance matrix
  cov_matrix <- cov(features, use = "complete.obs")
  
  # Check if covariance matrix is singular
  if (any(is.na(cov_matrix)) || det(cov_matrix) == 0 || !is.finite(det(cov_matrix))) {
    cat("Warning: Covariance matrix is singular. Using regularized version.\n")
    # Add small regularization
    cov_matrix <- cov_matrix + diag(1e-6, nrow(cov_matrix))
  }
  
  # Calculate Mahalanobis distance for each point
  distances <- mahalanobis(features, center = mu, cov = cov_matrix)
  
  return(list(distances = distances, mean = mu, cov = cov_matrix))
}

# Calculate Local Outlier Factor (LOF)
calculate_lof <- function(features, k = 5) {
  cat(sprintf("Calculating Local Outlier Factor (k = %d)...\n", k))
  
  # Use dbscan::lof
  lof_scores <- lof(features, k = k)
  
  return(lof_scores)
}

# Calculate Isolation Forest scores
calculate_isolation_forest <- function(features, n_trees = 100, nthreads = 1) {
  cat(sprintf("Calculating Isolation Forest scores (n_trees = %d, nthreads = %d)...\n", n_trees, nthreads))
  
  # Build isolation forest
  iso_forest <- isolation.forest(features, ntrees = n_trees, nthreads = nthreads)
  
  # Calculate anomaly scores (higher = more anomalous)
  scores <- predict(iso_forest, features)
  
  return(scores)
}

# Calculate feature contributions for Mahalanobis distance
calculate_mahalanobis_contributions <- function(feature_values, mean_vec, cov_matrix) {
  # For Mahalanobis: contribution = (x - μ)ᵀ Σ⁻¹
  diff <- feature_values - mean_vec
  
  # Calculate contribution vector
  tryCatch({
    inv_cov <- solve(cov_matrix)
    contributions <- as.vector(diff %*% inv_cov)
    names(contributions) <- names(feature_values)
    return(contributions)
  }, error = function(e) {
    # If inversion fails, use diagonal approximation
    diag_inv <- 1 / diag(cov_matrix)
    contributions <- diff * diag_inv
    names(contributions) <- names(feature_values)
    return(contributions)
  })
}

# Calculate feature contributions for LOF
calculate_lof_contributions <- function(feature_values, all_features, k = 5) {
  # Find k nearest neighbors
  distances <- sqrt(rowSums((all_features - matrix(feature_values, nrow = nrow(all_features), 
                                                   ncol = length(feature_values), byrow = TRUE))^2))
  k_nearest_idx <- order(distances)[2:(k+1)]  # Exclude self (distance = 0)
  k_nearest <- all_features[k_nearest_idx, , drop = FALSE]
  
  # Calculate mean of k nearest neighbors
  neighbor_mean <- colMeans(k_nearest, na.rm = TRUE)
  
  # Contribution is deviation from neighbor mean
  contributions <- feature_values - neighbor_mean
  names(contributions) <- names(feature_values)
  
  return(contributions)
}

# Calculate feature contributions for Isolation Forest
calculate_isolation_contributions <- function(feature_values, all_features, iso_forest) {
  # Use feature importance from isolation forest
  # For simplicity, calculate deviation from median
  feature_median <- apply(all_features, 2, median, na.rm = TRUE)
  contributions <- feature_values - feature_median
  names(contributions) <- names(feature_values)
  
  return(contributions)
}

# Generate outlier explanation
generate_explanation <- function(bout_row, feature_contributions, population_stats, method) {
  # Get top contributing features (by absolute value)
  abs_contributions <- abs(feature_contributions)
  top_indices <- order(abs_contributions, decreasing = TRUE)[1:min(5, length(abs_contributions))]
  
  top_features <- names(feature_contributions)[top_indices]
  top_contributions <- feature_contributions[top_indices]
  
  # Calculate z-scores for top features
  z_scores <- numeric(length(top_features))
  for (i in seq_along(top_features)) {
    feat_name <- top_features[i]
    if (feat_name %in% names(population_stats$means)) {
      z_scores[i] <- (bout_row[[feat_name]] - population_stats$means[[feat_name]]) / 
                     population_stats$sds[[feat_name]]
    } else {
      z_scores[i] <- NA
    }
  }
  
  # Build explanation text
  explanation_parts <- character(0)
  for (i in seq_along(top_features)) {
    if (!is.na(z_scores[i])) {
      direction <- if (z_scores[i] > 0) "high" else "low"
      explanation_parts <- c(explanation_parts, 
                            sprintf("%s %s (z-score: %.2f)", direction, top_features[i], z_scores[i]))
    }
  }
  
  explanation <- paste(explanation_parts, collapse = ", ")
  if (explanation == "") {
    explanation <- sprintf("Outlier detected by %s method", method)
  } else {
    explanation <- sprintf("Outlier due to unusually %s", explanation)
  }
  
  return(list(
    explanation = explanation,
    top_features = paste(top_features, collapse = ", "),
    z_scores = paste(sprintf("%.2f", z_scores), collapse = ", ")
  ))
}

# Main execution
main <- function() {
  cat("============================================================\n")
  cat("Outlier Detection Analysis\n")
  cat("============================================================\n\n")
  
  # Load features
  cat(sprintf("Loading features from: %s\n", opt$features))
  if (!file.exists(opt$features)) {
    stop(sprintf("Features file not found: %s", opt$features))
  }
  
  df <- read.csv(opt$features, stringsAsFactors = FALSE)
  cat(sprintf("Loaded %d bouts\n", nrow(df)))
  
  # Load cluster assignments if provided
  cluster_assignments <- NULL
  if (!is.null(opt$clusters) && file.exists(opt$clusters)) {
    cluster_assignments <- read.csv(opt$clusters, stringsAsFactors = FALSE)
    cat(sprintf("Loaded cluster assignments from: %s\n", opt$clusters))
  }
  
  # Prepare features
  cat("\nPreprocessing features...\n")
  exclude_cols <- c("bout_id", "video_name", "animal_id", "start_frame", "end_frame", "behavior")
  prep_result <- prepare_features(df, exclude_cols = exclude_cols)
  
  processed_df <- prep_result$processed_df
  feature_cols <- prep_result$feature_cols
  scaling_info <- prep_result$scaling_info
  
  cat(sprintf("After preprocessing: %d features\n", length(feature_cols)))
  
  # Extract feature matrix (keep original for explanations)
  feature_matrix_original <- as.matrix(processed_df[, feature_cols, drop = FALSE])
  feature_matrix_original[!is.finite(feature_matrix_original)] <- 0
  
  # Apply PCA if requested
  feature_matrix <- feature_matrix_original
  pca_result <- NULL
  if (opt$`use-pca`) {
    cat("\nApplying PCA reduction...\n")
    pca_result <- prcomp(feature_matrix_original, center = TRUE, scale. = TRUE)
    cumvar <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
    n_components <- which(cumvar >= opt$`pca-variance`)[1]
    if (is.na(n_components)) {
      n_components <- min(length(cumvar), ncol(feature_matrix_original))
    }
    feature_matrix <- pca_result$x[, 1:n_components, drop = FALSE]
    cat(sprintf("Reduced to %d PCA components (%.1f%% variance)\n", n_components, cumvar[n_components] * 100))
    cat("Note: Distances calculated in PCA space, but explanations use original features.\n")
  }
  
  # Create output directory and outlier_detection subdirectory
  dir.create(opt$`output-dir`, showWarnings = FALSE, recursive = TRUE)
  outlier_output_dir <- file.path(opt$`output-dir`, "outlier_detection")
  dir.create(outlier_output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Determine number of workers for parallel processing
  n_workers <- if (is.null(opt$workers)) {
    max(1, detectCores() - 1)
  } else {
    opt$workers
  }
  cat(sprintf("\nUsing %d parallel workers for computation\n", n_workers))
  
  # Calculate outlier scores for all three methods
  # Use PCA-reduced features if PCA was applied
  cat("\n============================================================\n")
  cat("Calculating Outlier Scores\n")
  cat("============================================================\n")
  
  # Mahalanobis distance
  mahal_result <- calculate_mahalanobis(feature_matrix)
  mahal_distances <- mahal_result$distances
  
  # LOF
  lof_scores <- calculate_lof(feature_matrix, k = 5)
  
  # Isolation Forest (uses parallel threads internally)
  iso_scores <- calculate_isolation_forest(feature_matrix, n_trees = 100, nthreads = n_workers)
  
  # Determine threshold for top outliers
  if (is.null(opt$`top-n`)) {
    n_outliers <- max(1, floor(nrow(df) * (1 - opt$percentile)))
  } else {
    n_outliers <- min(opt$`top-n`, nrow(df))
  }
  
  cat(sprintf("\nSelecting top %d outliers (%.1f%%)\n", n_outliers, (n_outliers/nrow(df))*100))
  
  # Identify outliers for each method
  mahal_outliers <- order(mahal_distances, decreasing = TRUE)[1:n_outliers]
  lof_outliers <- order(lof_scores, decreasing = TRUE)[1:n_outliers]
  iso_outliers <- order(iso_scores, decreasing = TRUE)[1:n_outliers]
  
  # Create outlier flags
  is_mahal_outlier <- seq_len(nrow(df)) %in% mahal_outliers
  is_lof_outlier <- seq_len(nrow(df)) %in% lof_outliers
  is_iso_outlier <- seq_len(nrow(df)) %in% iso_outliers
  consensus_outlier <- (is_mahal_outlier + is_lof_outlier + is_iso_outlier) >= 2
  
  # Calculate population statistics for explanations
  population_stats <- list(
    means = scaling_info$means,
    sds = scaling_info$sds
  )
  
  # Generate explanations for all outliers (parallelized)
  cat("\nGenerating outlier explanations...\n")
  
  all_outliers <- unique(c(mahal_outliers, lof_outliers, iso_outliers))
  
  # Pre-compute Mahalanobis on original features once (used by multiple outliers)
  orig_mahal_result <- calculate_mahalanobis(feature_matrix_original)
  
  # Function to process a single outlier
  process_outlier <- function(idx) {
    bout_row <- processed_df[idx, ]
    
    # For feature contributions, always use original feature space (not PCA)
    feature_values <- as.numeric(bout_row[, feature_cols, drop = FALSE])
    names(feature_values) <- feature_cols
    
    # Calculate contributions for each method using original feature space
    mahal_contrib <- if (idx %in% mahal_outliers) {
      calculate_mahalanobis_contributions(feature_values, orig_mahal_result$mean, orig_mahal_result$cov)
    } else {
      numeric(0)
    }
    
    lof_contrib <- if (idx %in% lof_outliers) {
      calculate_lof_contributions(feature_values, feature_matrix_original, k = 5)
    } else {
      numeric(0)
    }
    
    iso_contrib <- if (idx %in% iso_outliers) {
      calculate_isolation_contributions(feature_values, feature_matrix_original, NULL)
    } else {
      numeric(0)
    }
    
    # Use the method that flagged this outlier, or Mahalanobis as default
    if (idx %in% mahal_outliers && length(mahal_contrib) > 0) {
      contributions <- mahal_contrib
      method <- "Mahalanobis"
    } else if (idx %in% lof_outliers && length(lof_contrib) > 0) {
      contributions <- lof_contrib
      method <- "LOF"
    } else if (idx %in% iso_outliers && length(iso_contrib) > 0) {
      contributions <- iso_contrib
      method <- "Isolation Forest"
    } else {
      contributions <- mahal_contrib
      method <- "Mahalanobis"
    }
    
    # Generate explanation
    expl_result <- generate_explanation(bout_row, contributions, population_stats, method)
    
    # Get cluster assignment if available
    cluster_id <- NA
    distance_to_centroid <- NA
    if (!is.null(cluster_assignments)) {
      bout_id <- bout_row$bout_id
      cluster_row <- cluster_assignments[cluster_assignments$bout_id == bout_id, ]
      if (nrow(cluster_row) > 0) {
        cluster_id <- cluster_row$cluster[1]
      }
    }
    
    explanation_row <- list(
      bout_id = bout_row$bout_id,
      video_name = bout_row$video_name,
      animal_id = bout_row$animal_id,
      start_frame = bout_row$start_frame,
      end_frame = bout_row$end_frame,
      outlier_score_mahalanobis = mahal_distances[idx],
      outlier_score_lof = lof_scores[idx],
      outlier_score_isolation = iso_scores[idx],
      is_outlier_mahalanobis = is_mahal_outlier[idx],
      is_outlier_lof = is_lof_outlier[idx],
      is_outlier_isolation = is_iso_outlier[idx],
      consensus_outlier = consensus_outlier[idx],
      top_contributing_features = expl_result$top_features,
      feature_deviations = expl_result$z_scores,
      cluster_assignment = cluster_id,
      distance_to_centroid = distance_to_centroid,
      explanation = expl_result$explanation
    )
    
    # Store feature contributions
    contribution_rows <- list()
    for (feat_name in names(contributions)) {
      feat_value <- if (feat_name %in% names(bout_row)) bout_row[[feat_name]] else NA
      pop_mean <- if (feat_name %in% names(population_stats$means)) population_stats$means[[feat_name]] else NA
      pop_sd <- if (feat_name %in% names(population_stats$sds)) population_stats$sds[[feat_name]] else NA
      z_score <- if (!is.na(pop_mean) && !is.na(pop_sd) && pop_sd > 0) {
        (feat_value - pop_mean) / pop_sd
      } else {
        NA
      }
      
      contribution_rows[[length(contribution_rows) + 1]] <- list(
        bout_id = bout_row$bout_id,
        feature_name = feat_name,
        feature_value = feat_value,
        population_mean = pop_mean,
        population_std = pop_sd,
        z_score = z_score,
        contribution_rank = which(order(abs(contributions), decreasing = TRUE) == which(names(contributions) == feat_name))
      )
    }
    
    return(list(explanation = explanation_row, contributions = contribution_rows))
  }
  
  # Process outliers in parallel
  if (length(all_outliers) > 1 && n_workers > 1) {
    cat(sprintf("Processing %d outliers in parallel using %d workers...\n", length(all_outliers), n_workers))
    cl <- makeCluster(n_workers)
    on.exit(stopCluster(cl), add = TRUE)
    
    # Source utility functions on workers
    clusterEvalQ(cl, {
      source("analysis_r/utils/data_preprocessing.R")
    })
    
    # Export all necessary variables and functions to workers
    # Use the current environment to access variables from main() function
    clusterExport(cl, varlist = c(
      "processed_df", "feature_cols", "feature_matrix_original", 
      "mahal_outliers", "lof_outliers", "iso_outliers",
      "orig_mahal_result", "population_stats", "cluster_assignments",
      "mahal_distances", "lof_scores", "iso_scores",
      "is_mahal_outlier", "is_lof_outlier", "is_iso_outlier", "consensus_outlier",
      "calculate_mahalanobis_contributions", 
      "calculate_lof_contributions",
      "calculate_isolation_contributions", 
      "generate_explanation",
      "process_outlier"
    ), envir = environment())
    
    results <- parLapply(cl, all_outliers, process_outlier)
  } else {
    # Sequential processing for small datasets or single worker
    results <- lapply(all_outliers, process_outlier)
  }
  
  # Combine results
  explanations_list <- lapply(results, function(x) x$explanation)
  feature_contributions_list <- do.call(c, lapply(results, function(x) x$contributions))
  
  # Convert to data frames
  explanations_df <- do.call(rbind, lapply(explanations_list, function(x) data.frame(x, stringsAsFactors = FALSE)))
  contributions_df <- do.call(rbind, lapply(feature_contributions_list, function(x) data.frame(x, stringsAsFactors = FALSE)))
  
  # Save results
  cat("\nSaving results...\n")
  
  # Outlier explanations
  explanations_file <- file.path(outlier_output_dir, "outlier_explanations.csv")
  write.csv(explanations_df, explanations_file, row.names = FALSE)
  cat(sprintf("Outlier explanations saved to: %s\n", explanations_file))
  
  # Feature contributions
  contributions_file <- file.path(outlier_output_dir, "outlier_feature_contributions.csv")
  write.csv(contributions_df, contributions_file, row.names = FALSE)
  cat(sprintf("Feature contributions saved to: %s\n", contributions_file))
  
  # Method-specific outlier lists
  mahal_outliers_df <- explanations_df[explanations_df$is_outlier_mahalanobis, ]
  write.csv(mahal_outliers_df[, c("bout_id", "video_name", "animal_id", "start_frame", "end_frame")], 
            file.path(outlier_output_dir, "outliers_mahalanobis.csv"), row.names = FALSE)
  
  lof_outliers_df <- explanations_df[explanations_df$is_outlier_lof, ]
  write.csv(lof_outliers_df[, c("bout_id", "video_name", "animal_id", "start_frame", "end_frame")], 
            file.path(outlier_output_dir, "outliers_lof.csv"), row.names = FALSE)
  
  iso_outliers_df <- explanations_df[explanations_df$is_outlier_isolation, ]
  write.csv(iso_outliers_df[, c("bout_id", "video_name", "animal_id", "start_frame", "end_frame")], 
            file.path(outlier_output_dir, "outliers_isolation_forest.csv"), row.names = FALSE)
  
  consensus_outliers_df <- explanations_df[explanations_df$consensus_outlier, ]
  write.csv(consensus_outliers_df[, c("bout_id", "video_name", "animal_id", "start_frame", "end_frame")], 
            file.path(outlier_output_dir, "outliers_consensus.csv"), row.names = FALSE)
  
  cat("\n============================================================\n")
  cat("Outlier detection complete!\n")
  cat(sprintf("  - Mahalanobis outliers: %d\n", sum(is_mahal_outlier)))
  cat(sprintf("  - LOF outliers: %d\n", sum(is_lof_outlier)))
  cat(sprintf("  - Isolation Forest outliers: %d\n", sum(is_iso_outlier)))
  cat(sprintf("  - Consensus outliers: %d\n", sum(consensus_outlier)))
}

# Run main
main()

