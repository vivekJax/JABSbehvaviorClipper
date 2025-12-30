#!/usr/bin/env Rscript
# Fix R environment issues
tryCatch({ options(editor = "vim") }, error = function(e) { tryCatch({ options(editor = NULL) }, error = function(e2) BoutAnalysisScripts/scripts/core/find_outliers.R) })
# Fix R environment issues
options(editor = NULL)
options(defaultPackages = c("datasets", "utils", "grDevices", "graphics", "stats", "methods"))
# Find outlier behavior bouts using aggregate distance metrics.
#
# This script:
# 1. Loads bout features
# 2. Calculates pairwise distances between all bouts
# 3. Identifies outliers using aggregate distance metrics
# 4. Generates videos for outliers using the same standards as cluster videos
#
# Video Standards (inherited from generate_bouts_video.py):
#   - Codec: libx264 + aac
#   - Frame rate: 30 fps
#   - Preset: fast
#   - Font: /System/Library/Fonts/Helvetica.ttc (fontsize=20)
#   - Bounding boxes: Yellow outline (t=3, color=yellow@1.0)
#   - Text overlay: Bottom center with black semi-transparent background
#   - Default workers: CPU cores - 1 (leaves one core free)
#
# Outlier Detection Methods:
#   - mean_distance: Mean distance to all other bouts (default)
#   - median_distance: Median distance to all other bouts
#   - max_distance: Maximum distance to any other bout
#   - knn_distance: Mean distance to k nearest neighbors (k = sqrt(n))

# Add default library paths (user library first)
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(jsonlite)
  library(ggplot2)
  library(Rtsne)
  library(gridExtra)
  library(MASS)  # For ginv (pseudo-inverse) in Mahalanobis distance
})

# Parse command line arguments
option_list <- list(
  make_option(c("-f", "--features"), type="character", default="bout_features.csv",
              help="Input CSV file with bout features (default: bout_features.csv)"),
  make_option(c("-m", "--method"), type="character", default="mean_distance",
              help="Outlier detection method: mean_distance, median_distance, max_distance, or isolation_forest (default: mean_distance)"),
  make_option(c("-t", "--threshold"), type="character", default="auto",
              help="Outlier threshold: 'auto' (top 5%%), 'topN' (top N), or numeric percentile (default: auto)"),
  make_option(c("-n", "--top-n"), type="integer", default=NULL,
              help="Number of top outliers to select (used with threshold='topN')"),
  make_option(c("-d", "--distance-metric"), type="character", default="euclidean",
              help="Distance metric: euclidean, manhattan, cosine, or mahalanobis (default: euclidean)"),
  make_option(c("--use-pca"), action="store_true", default=FALSE,
              help="Use PCA-reduced dimensions for distance calculation (removes correlation)"),
  make_option(c("--pca-variance"), type="numeric", default=0.95,
              help="Proportion of variance to retain in PCA (default: 0.95)"),
  make_option(c("-s", "--scale-method"), type="character", default="standard",
              help="Feature scaling method: standard, minmax, or robust (default: standard)"),
  make_option(c("-o", "--output-dir"), type="character", default="outlier_videos",
              help="Output directory for outlier videos"),
  make_option(c("-v", "--video-dir"), type="character", default=".",
              help="Directory containing video files"),
  make_option(c("--video-clipper"), type="character", default="generate_bouts_video.py",
              help="Path to Python video clipper script"),
  make_option(c("--behavior"), type="character", default="turn_left",
              help="Behavior name for video clipping"),
  make_option(c("--workers"), type="integer", default=NULL,
              help="Number of parallel workers for video clipping"),
  make_option(c("--keep-temp"), action="store_true", default=FALSE,
              help="Keep temporary clip files"),
  make_option(c("--verbose"), action="store_true", default=FALSE,
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

# Handle missing values and scale
prepare_features <- function(df) {
  separated <- separate_metadata_and_features(df)
  metadata_df <- separated$metadata
  features_df <- separated$features
  
  # Drop columns with all NA
  features_df <- features_df[, colSums(!is.na(features_df)) > 0, drop=FALSE]
  
  # Fill missing values with mean (preserve all rows)
  for (col in colnames(features_df)) {
    if (any(is.na(features_df[[col]]))) {
      col_mean <- mean(features_df[[col]], na.rm=TRUE)
      if (!is.na(col_mean)) {
        features_df[[col]][is.na(features_df[[col]])] <- col_mean
      } else {
        features_df[[col]][is.na(features_df[[col]])] <- 0
      }
    }
  }
  
  # Scale features
  X <- as.matrix(features_df)
  X[is.infinite(X)] <- 0
  X[is.nan(X)] <- 0
  
  # Remove constant/zero variance columns
  feature_vars <- apply(X, 2, var, na.rm=TRUE)
  non_constant_cols <- which(feature_vars > 1e-10)
  X <- X[, non_constant_cols, drop=FALSE]
  feature_names_filtered <- colnames(features_df)[non_constant_cols]
  
  if (opt$`scale-method` == "standard") {
    X_scaled <- scale(X)
  } else if (opt$`scale-method` == "minmax") {
    min_vals <- apply(X, 2, min, na.rm=TRUE)
    max_vals <- apply(X, 2, max, na.rm=TRUE)
    X_scaled <- sweep(sweep(X, 2, min_vals, "-"), 2, max_vals - min_vals, "/")
  } else if (opt$`scale-method` == "robust") {
    median_vals <- apply(X, 2, median, na.rm=TRUE)
    mad_vals <- apply(X, 2, mad, na.rm=TRUE)
    X_scaled <- sweep(sweep(X, 2, median_vals, "-"), 2, mad_vals, "/")
  }
  
  X_scaled[!is.finite(X_scaled)] <- 0
  
  return(list(metadata=metadata_df, features=X_scaled, feature_names=feature_names_filtered))
}

# Apply PCA dimensionality reduction
apply_pca_reduction <- function(X, variance_threshold = 0.95) {
  # Perform PCA
  pca_result <- prcomp(X, scale.=FALSE, center=FALSE)  # Already scaled
  
  # Calculate cumulative variance explained
  variance_explained <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
  
  # Find number of components needed to retain variance threshold
  n_components <- which(variance_explained >= variance_threshold)[1]
  if (is.na(n_components)) {
    n_components <- length(variance_explained)
  }
  
  # Extract reduced dimensions
  X_reduced <- pca_result$x[, 1:n_components, drop=FALSE]
  
  cat(sprintf("PCA reduction: %d features -> %d components (%.1f%% variance retained)\n",
             ncol(X), n_components, variance_explained[n_components] * 100))
  
  return(list(
    X_reduced = X_reduced,
    pca_result = pca_result,
    n_components = n_components,
    variance_explained = variance_explained[n_components]
  ))
}

# Calculate distance matrix
calculate_distance_matrix <- function(X, metric = "euclidean", use_pca = FALSE, pca_variance = 0.95) {
  # Apply PCA reduction if requested
  if (use_pca) {
    pca_info <- apply_pca_reduction(X, pca_variance)
    X <- pca_info$X_reduced
  }
  
  if (metric == "euclidean") {
    dist_matrix <- as.matrix(dist(X, method="euclidean"))
  } else if (metric == "manhattan") {
    dist_matrix <- as.matrix(dist(X, method="manhattan"))
  } else if (metric == "cosine") {
    # Cosine distance = 1 - cosine similarity
    X_norm <- X / sqrt(rowSums(X^2))
    X_norm[!is.finite(X_norm)] <- 0
    cosine_sim <- X_norm %*% t(X_norm)
    dist_matrix <- 1 - cosine_sim
    diag(dist_matrix) <- 0  # Distance to self is 0
  } else if (metric == "mahalanobis") {
    # Mahalanobis distance accounts for covariance structure
    # D(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
    # where S is the covariance matrix
    
    # Calculate covariance matrix
    cov_matrix <- cov(X)
    
    # Check if covariance matrix is invertible
    if (det(cov_matrix) < 1e-10) {
      cat("Warning: Covariance matrix is near-singular. Using pseudo-inverse.\n")
      # Use pseudo-inverse for near-singular matrices
      cov_inv <- MASS::ginv(cov_matrix)
    } else {
      cov_inv <- solve(cov_matrix)
    }
    
    # Calculate Mahalanobis distance for all pairs
    n <- nrow(X)
    dist_matrix <- matrix(0, nrow=n, ncol=n)
    
    for (i in 1:(n-1)) {
      for (j in (i+1):n) {
        diff <- X[i, ] - X[j, ]
        dist_sq <- as.numeric(t(diff) %*% cov_inv %*% diff)
        dist_matrix[i, j] <- sqrt(max(0, dist_sq))  # Ensure non-negative
        dist_matrix[j, i] <- dist_matrix[i, j]  # Symmetric
      }
    }
    
    cat("Calculated Mahalanobis distance matrix (accounts for feature correlations)\n")
  } else {
    stop(sprintf("Unknown distance metric: %s", metric))
  }
  
  return(dist_matrix)
}

# Calculate aggregate distance metrics for each bout
calculate_aggregate_distances <- function(dist_matrix, method = "mean_distance") {
  n <- nrow(dist_matrix)
  aggregate_scores <- numeric(n)
  
  if (method == "mean_distance") {
    # Mean distance to all other points
    for (i in 1:n) {
      aggregate_scores[i] <- mean(dist_matrix[i, -i], na.rm=TRUE)
    }
  } else if (method == "median_distance") {
    # Median distance to all other points
    for (i in 1:n) {
      aggregate_scores[i] <- median(dist_matrix[i, -i], na.rm=TRUE)
    }
  } else if (method == "max_distance") {
    # Maximum distance to any other point
    for (i in 1:n) {
      aggregate_scores[i] <- max(dist_matrix[i, -i], na.rm=TRUE)
    }
  } else if (method == "knn_distance") {
    # Mean distance to k nearest neighbors (k = sqrt(n))
    k <- max(1, floor(sqrt(n)))
    for (i in 1:n) {
      distances <- sort(dist_matrix[i, -i])
      aggregate_scores[i] <- mean(distances[1:min(k, length(distances))], na.rm=TRUE)
    }
  } else {
    stop(sprintf("Unknown method: %s", method))
  }
  
  return(aggregate_scores)
}

# Identify outliers based on threshold
identify_outliers <- function(aggregate_scores, threshold = "auto", top_n = NULL) {
  if (threshold == "auto") {
    # Use top 5% as outliers
    threshold_value <- quantile(aggregate_scores, 0.95, na.rm=TRUE)
    outlier_indices <- which(aggregate_scores >= threshold_value)
  } else if (threshold == "topN") {
    if (is.null(top_n)) {
      stop("--top-n must be specified when using threshold='topN'")
    }
    # Select top N outliers
    sorted_indices <- order(aggregate_scores, decreasing=TRUE)
    outlier_indices <- sorted_indices[1:min(top_n, length(sorted_indices))]
  } else {
    # Numeric percentile threshold
    threshold_value <- as.numeric(threshold)
    if (threshold_value < 0 || threshold_value > 1) {
      stop("Numeric threshold must be between 0 and 1 (percentile)")
    }
    threshold_value <- quantile(aggregate_scores, threshold_value, na.rm=TRUE)
    outlier_indices <- which(aggregate_scores >= threshold_value)
  }
  
  return(outlier_indices)
}

# Create annotation JSON files for outliers (one per video)
create_outlier_annotations <- function(df, outlier_indices, annotations_dir) {
  outlier_bouts <- df[outlier_indices, , drop=FALSE]
  
  if (nrow(outlier_bouts) == 0) {
    cat("No outlier bouts found\n")
    return(character(0))
  }
  
  # Group by video
  videos <- unique(outlier_bouts$video_name)
  annotation_files <- character(0)
  
  for (video_name in videos) {
    video_bouts <- outlier_bouts[outlier_bouts$video_name == video_name, , drop=FALSE]
    
    # Group by animal
    animals <- unique(video_bouts$animal_id)
    labels <- list()
    
    for (animal_id in animals) {
      animal_bouts <- video_bouts[video_bouts$animal_id == animal_id, , drop=FALSE]
      
      # Create behavior bouts list
      behavior_bouts <- list()
      
      for (i in seq_len(nrow(animal_bouts))) {
        bout <- list(
          start = as.integer(animal_bouts$start_frame[i]),
          end = as.integer(animal_bouts$end_frame[i]),
          present = TRUE
        )
        behavior_bouts[[length(behavior_bouts) + 1]] <- bout
      }
      
      labels[[as.character(animal_id)]] <- list()
      labels[[as.character(animal_id)]][[opt$behavior]] <- behavior_bouts
    }
    
    # Create annotation structure
    annotation_data <- list(
      file = as.character(video_name)[1],
      labels = labels
    )
    
    # Create filename
    video_basename <- tools::file_path_sans_ext(video_name)[1]
    json_filename <- paste0(video_basename, ".json")
    json_file <- file.path(annotations_dir, json_filename)
    
    # Write JSON file
    write_json(annotation_data, json_file, pretty=TRUE, auto_unbox=TRUE)
    annotation_files <- c(annotation_files, json_file)
    
    cat(sprintf("  Created annotation: %s (%d bouts)\n", json_filename, nrow(video_bouts)))
  }
  
  cat(sprintf("Created %d annotation files for outliers (total %d bouts)\n", 
             length(annotation_files), nrow(outlier_bouts)))
  
  return(annotation_files)
}

# Create visualizations to explain outliers
create_outlier_plots <- function(df, X_scaled, aggregate_scores, outlier_indices, output_dir) {
  # Prepare data for plotting
  plot_df <- data.frame(
    bout_id = df$bout_id,
    aggregate_distance = aggregate_scores,
    is_outlier = 1:nrow(df) %in% outlier_indices,
    video_name = df$video_name,
    animal_id = df$animal_id
  )
  
  # Plot 1: Distance distribution histogram
  p1 <- ggplot(plot_df, aes(x=aggregate_distance, fill=is_outlier)) +
    geom_histogram(bins=30, alpha=0.7, position="identity") +
    scale_fill_manual(values=c("FALSE"="gray70", "TRUE"="red"), 
                     labels=c("FALSE"="Normal", "TRUE"="Outlier"),
                     name="Type") +
    labs(
      x = "Aggregate Distance",
      y = "Number of Bouts",
      title = "Distribution of Aggregate Distances",
      subtitle = "Outliers are bouts with unusually high distances to other bouts",
      caption = "Interpretation: Histogram shows the distribution of aggregate distances (mean/median distance to all other bouts). Outliers (red) appear in the right tail with high distances, indicating they are dissimilar to most other bouts. A clear separation suggests good outlier detection."
    ) +
    theme_minimal() +
    theme(plot.caption = element_text(hjust=0, size=8, margin=margin(t=10))) +
    theme(legend.position="right")
  
  # Add vertical line at threshold
  threshold_value <- min(aggregate_scores[outlier_indices], na.rm=TRUE)
  p1 <- p1 + geom_vline(xintercept=threshold_value, linetype="dashed", color="red", linewidth=1) +
    annotate("text", x=threshold_value, y=Inf, label="Outlier\nThreshold", 
             vjust=1.5, hjust=0, color="red", size=3)
  
  # Plot 2: Distance vs index (sorted)
  plot_df_sorted <- plot_df[order(plot_df$aggregate_distance), ]
  plot_df_sorted$rank <- 1:nrow(plot_df_sorted)
  
  p2 <- ggplot(plot_df_sorted, aes(x=rank, y=aggregate_distance, color=is_outlier)) +
    geom_point(alpha=0.7, size=2) +
    scale_color_manual(values=c("FALSE"="gray70", "TRUE"="red"),
                      labels=c("FALSE"="Normal", "TRUE"="Outlier"),
                      name="Type") +
    labs(
      x = "Bout Rank (sorted by distance)",
      y = "Aggregate Distance",
      title = "Outliers Stand Out in Distance Ranking",
      subtitle = "Outliers have the highest aggregate distances",
      caption = "Interpretation: Bouts are ranked by aggregate distance. Outliers (red) appear at the top with highest distances. A clear jump in distance values indicates a natural threshold separating outliers from normal bouts."
    ) +
    theme_minimal() +
    theme(plot.caption = element_text(hjust=0, size=8, margin=margin(t=10))) +
    theme(legend.position="right")
  
  # Plot 3: PCA visualization
  if (nrow(X_scaled) >= 5) {
    tryCatch({
      pca_result <- prcomp(X_scaled, scale.=FALSE)
      pca_df <- data.frame(
        PC1 = pca_result$x[, 1],
        PC2 = pca_result$x[, 2],
        aggregate_distance = aggregate_scores,
        is_outlier = 1:nrow(X_scaled) %in% outlier_indices,
        animal_id = df$animal_id
      )
      
      p3 <- ggplot(pca_df, aes(x=PC1, y=PC2, color=is_outlier, size=aggregate_distance)) +
        geom_point(alpha=0.7) +
        scale_color_manual(values=c("FALSE"="gray70", "TRUE"="red"),
                          labels=c("FALSE"="Normal", "TRUE"="Outlier"),
                          name="Type") +
        scale_size_continuous(name="Distance", range=c(1, 4)) +
        labs(
          x = sprintf("PC1 (%.1f%% variance)", summary(pca_result)$importance[2, 1] * 100),
          y = sprintf("PC2 (%.1f%% variance)", summary(pca_result)$importance[2, 2] * 100),
          title = "Outliers in Feature Space (PCA)",
          subtitle = "Outliers are often separated from the main cluster",
          caption = "Interpretation: PCA visualization of all bouts. Outliers (red) are typically separated from the main cluster of normal bouts (gray). Point size indicates aggregate distance - larger points are more extreme outliers. Outliers may represent unusual behavior patterns or data quality issues."
        ) +
        theme_minimal() +
        theme(legend.position="right",
              plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
      
      # Save PCA plot
      ggsave(file.path(output_dir, "outliers_pca.png"), p3, width=10, height=8, dpi=300)
      cat("  Saved PCA plot: outliers_pca.png\n")
    }, error = function(e) {
      cat(sprintf("  Could not create PCA plot: %s\n", e$message))
    })
  }
  
  # Plot 4: t-SNE visualization (if enough samples)
  if (nrow(X_scaled) >= 5 && nrow(X_scaled) <= 1000) {
    tryCatch({
      perplexity <- min(30, max(5, floor((nrow(X_scaled) - 1) / 3)))
      tsne_result <- Rtsne(X_scaled, dims=2, perplexity=perplexity, verbose=FALSE)
      
      tsne_df <- data.frame(
        tSNE1 = tsne_result$Y[, 1],
        tSNE2 = tsne_result$Y[, 2],
        aggregate_distance = aggregate_scores,
        is_outlier = 1:nrow(X_scaled) %in% outlier_indices,
        animal_id = df$animal_id
      )
      
      p4 <- ggplot(tsne_df, aes(x=tSNE1, y=tSNE2, color=is_outlier, size=aggregate_distance)) +
        geom_point(alpha=0.7) +
        scale_color_manual(values=c("FALSE"="gray70", "TRUE"="red"),
                          labels=c("FALSE"="Normal", "TRUE"="Outlier"),
                          name="Type") +
        scale_size_continuous(name="Distance", range=c(1, 4)) +
        labs(
          x = "t-SNE Dimension 1",
          y = "t-SNE Dimension 2",
          title = "Outliers in Feature Space (t-SNE)",
          subtitle = "Non-linear view showing outliers separated from main group",
          caption = "Interpretation: t-SNE provides a non-linear view of the feature space. Outliers (red) should appear separated from the main group of normal bouts. This view often reveals outlier structure better than PCA, especially for non-linear patterns."
        ) +
        theme_minimal() +
        theme(legend.position="right",
              plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
      
      # Save t-SNE plot
      ggsave(file.path(output_dir, "outliers_tsne.png"), p4, width=10, height=8, dpi=300)
      cat("  Saved t-SNE plot: outliers_tsne.png\n")
    }, error = function(e) {
      cat(sprintf("  Could not create t-SNE plot: %s\n", e$message))
    })
  }
  
  # Plot 5: Box plot comparing outlier vs normal distances
  p5 <- ggplot(plot_df, aes(x=is_outlier, y=aggregate_distance, fill=is_outlier)) +
    geom_boxplot(alpha=0.7) +
    geom_jitter(width=0.2, alpha=0.5, size=1) +
    scale_fill_manual(values=c("FALSE"="gray70", "TRUE"="red"),
                     labels=c("FALSE"="Normal", "TRUE"="Outlier"),
                     name="Type") +
    scale_x_discrete(labels=c("FALSE"="Normal\nBouts", "TRUE"="Outlier\nBouts")) +
    labs(
      x = "Bout Type",
      y = "Aggregate Distance",
      title = "Distance Comparison: Outliers vs Normal Bouts",
      subtitle = "Outliers have significantly higher aggregate distances",
      caption = "Interpretation: Boxplot comparing aggregate distances between normal and outlier bouts. Outliers should have clearly higher median and range of distances. Minimal overlap between boxes indicates good separation. Jittered points show individual bout distances."
    ) +
    theme_minimal() +
    theme(legend.position="none",
          plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
  
  # Plot 6: Top features that distinguish outliers
  if (length(outlier_indices) > 0 && length(outlier_indices) < nrow(X_scaled)) {
    # Calculate mean features for outliers vs normal
    outlier_features <- colMeans(X_scaled[outlier_indices, , drop=FALSE], na.rm=TRUE)
    normal_indices <- setdiff(1:nrow(X_scaled), outlier_indices)
    normal_features <- colMeans(X_scaled[normal_indices, , drop=FALSE], na.rm=TRUE)
    
    # Calculate difference
    feature_diff <- abs(outlier_features - normal_features)
    top_features <- names(sort(feature_diff, decreasing=TRUE))[1:min(10, length(feature_diff))]
    
    feature_comp_df <- data.frame(
      feature = rep(top_features, 2),
      value = c(outlier_features[top_features], normal_features[top_features]),
      type = rep(c("Outlier", "Normal"), each=length(top_features))
    )
    
    p6 <- ggplot(feature_comp_df, aes(x=reorder(feature, value), y=value, fill=type)) +
      geom_bar(stat="identity", position="dodge", alpha=0.7) +
      scale_fill_manual(values=c("Outlier"="red", "Normal"="gray70"), name="Type") +
      coord_flip() +
      labs(
        x = "Feature",
        y = "Mean Scaled Value",
        title = "Top Features Distinguishing Outliers",
        subtitle = "Features with largest difference between outliers and normal bouts",
        caption = "Interpretation: Bar chart comparing mean feature values between outliers and normal bouts. Features are ordered by the magnitude of difference. Large differences (red vs gray bars) indicate which features best distinguish outliers. This helps understand what makes outliers unusual."
      ) +
      theme_minimal() +
      theme(legend.position="right",
            plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
    
    # Save feature comparison plot
    ggsave(file.path(output_dir, "outliers_features.png"), p6, width=12, height=8, dpi=300)
    cat("  Saved feature comparison plot: outliers_features.png\n")
  }
  
  # Combine and save main plots
  combined_plot <- gridExtra::grid.arrange(p1, p2, p5, ncol=2, nrow=2)
  ggsave(file.path(output_dir, "outliers_analysis.png"), combined_plot, width=16, height=12, dpi=300)
  cat("  Saved combined analysis plot: outliers_analysis.png\n")
  
  # Create summary statistics plot
  summary_stats <- data.frame(
    Type = c("Normal Bouts", "Outlier Bouts"),
    Count = c(sum(!plot_df$is_outlier), sum(plot_df$is_outlier)),
    Mean_Distance = c(
      mean(plot_df$aggregate_distance[!plot_df$is_outlier], na.rm=TRUE),
      mean(plot_df$aggregate_distance[plot_df$is_outlier], na.rm=TRUE)
    ),
    Median_Distance = c(
      median(plot_df$aggregate_distance[!plot_df$is_outlier], na.rm=TRUE),
      median(plot_df$aggregate_distance[plot_df$is_outlier], na.rm=TRUE)
    )
  )
  
  p7 <- ggplot(summary_stats, aes(x=Type, y=Mean_Distance, fill=Type)) +
    geom_bar(stat="identity", alpha=0.7) +
    geom_text(aes(label=sprintf("%.2f", Mean_Distance)), vjust=-0.5) +
    scale_fill_manual(values=c("Normal Bouts"="gray70", "Outlier Bouts"="red")) +
    labs(
      x = "Bout Type",
      y = "Mean Aggregate Distance",
      title = "Summary: Why These Are Outliers",
      subtitle = sprintf("Outliers have %.1fx higher mean distance than normal bouts",
                        summary_stats$Mean_Distance[2] / summary_stats$Mean_Distance[1]),
      caption = "Interpretation: Summary bar chart showing the mean aggregate distance for normal vs outlier bouts. The fold-change indicates how much more distant outliers are on average. Higher fold-changes indicate clearer outlier separation."
    ) +
    theme_minimal() +
    theme(legend.position="none",
          plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
  
  ggsave(file.path(output_dir, "outliers_summary.png"), p7, width=10, height=6, dpi=300)
  cat("  Saved summary plot: outliers_summary.png\n")
  
  cat("Visualization complete!\n")
}

# Generate video for outliers
generate_outlier_video <- function(annotation_files) {
  output_path <- file.path(opt$`output-dir`, "outliers.mp4")
  
  temp_annotations_dir <- dirname(annotation_files[1])
  
  # Build command - use same standards as generate_bouts_video.py
  if (is.null(opt$workers)) {
    cpu_cores_str <- system("python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())'", intern=TRUE)
    cpu_cores <- as.integer(cpu_cores_str[1])
    default_workers <- max(1, cpu_cores - 1)
    workers_arg <- sprintf("--workers %d", default_workers)
  } else {
    workers_arg <- sprintf("--workers %d", opt$workers)
  }
  
  cmd <- sprintf("python3 %s --behavior %s --annotations-dir %s --video-dir %s --output %s %s",
                opt$`video-clipper`,
                opt$behavior,
                temp_annotations_dir,
                opt$`video-dir`,
                output_path,
                workers_arg)
  
  if (opt$`keep-temp`) {
    cmd <- paste(cmd, "--keep-temp")
  }
  
  if (opt$verbose) {
    cmd <- paste(cmd, "--verbose")
  }
  
  cat(sprintf("Generating outlier video (%d annotation files)...\n", length(annotation_files)))
  
  if (opt$verbose) {
    cat(sprintf("Command: %s\n", cmd))
  }
  
  result <- system(cmd, intern=FALSE)
  
  if (result == 0) {
    cat(sprintf("✓ Successfully created outlier video: %s\n", output_path))
    return(output_path)
  } else {
    cat(sprintf("✗ Failed to create outlier video (exit code: %d)\n", result))
    return(NULL)
  }
}

# Main execution
main <- function() {
  cat(sprintf("Loading features from %s\n", opt$features))
  df <- read.csv(opt$features, stringsAsFactors=FALSE)
  
  if (nrow(df) == 0) {
    stop("Input file is empty")
  }
  
  cat(sprintf("Loaded %d bouts\n", nrow(df)))
  
  # Prepare features
  prepared <- prepare_features(df)
  metadata_df <- prepared$metadata
  X_scaled <- prepared$features
  
  cat(sprintf("Prepared %d samples with %d features\n", nrow(X_scaled), ncol(X_scaled)))
  
  # Calculate distance matrix
  if (opt$`use-pca`) {
    cat(sprintf("Calculating %s distance matrix on PCA-reduced dimensions...\n", opt$`distance-metric`))
  } else {
    cat(sprintf("Calculating %s distance matrix...\n", opt$`distance-metric`))
  }
  dist_matrix <- calculate_distance_matrix(X_scaled, 
                                          metric=opt$`distance-metric`,
                                          use_pca=opt$`use-pca`,
                                          pca_variance=opt$`pca-variance`)
  
  # Calculate aggregate distances
  cat(sprintf("Calculating aggregate distances using method: %s\n", opt$method))
  aggregate_scores <- calculate_aggregate_distances(dist_matrix, method=opt$method)
  
  # Identify outliers
  cat(sprintf("Identifying outliers (threshold: %s)...\n", opt$threshold))
  outlier_indices <- identify_outliers(aggregate_scores, threshold=opt$threshold, top_n=opt$`top-n`)
  
  cat(sprintf("Found %d outliers out of %d total bouts\n", length(outlier_indices), nrow(df)))
  
  if (length(outlier_indices) == 0) {
    cat("No outliers found. Exiting.\n")
    return
  }
  
  # Print outlier statistics
  cat("\nOutlier Statistics:\n")
  cat(sprintf("  Aggregate distance range: [%.3f, %.3f]\n", 
             min(aggregate_scores, na.rm=TRUE), max(aggregate_scores, na.rm=TRUE)))
  cat(sprintf("  Outlier distance range: [%.3f, %.3f]\n",
             min(aggregate_scores[outlier_indices], na.rm=TRUE),
             max(aggregate_scores[outlier_indices], na.rm=TRUE)))
  cat(sprintf("  Mean outlier distance: %.3f\n",
             mean(aggregate_scores[outlier_indices], na.rm=TRUE)))
  
  # Create output directory
  dir.create(opt$`output-dir`, showWarnings=FALSE, recursive=TRUE)
  temp_ann_dir <- file.path(opt$`output-dir`, "temp_annotations", "outliers")
  dir.create(temp_ann_dir, showWarnings=FALSE, recursive=TRUE)
  
  # Create annotation files
  cat("\nCreating annotation files for outliers...\n")
  annotation_files <- create_outlier_annotations(df, outlier_indices, temp_ann_dir)
  
  if (length(annotation_files) == 0) {
    cat("No annotation files created. Exiting.\n")
    return
  }
  
  # Generate video
  cat("\nGenerating outlier video...\n")
  video_path <- generate_outlier_video(annotation_files)
  
  # Save outlier information
  outlier_df <- df[outlier_indices, ]
  outlier_df$aggregate_distance <- aggregate_scores[outlier_indices]
  outlier_df <- outlier_df[order(outlier_df$aggregate_distance, decreasing=TRUE), ]
  
  outlier_csv <- file.path(opt$`output-dir`, "outliers.csv")
  write.csv(outlier_df, outlier_csv, row.names=FALSE)
  cat(sprintf("Saved outlier information to %s\n", outlier_csv))
  
  # Create visualizations
  cat("\nCreating visualizations...\n")
  create_outlier_plots(df, X_scaled, aggregate_scores, outlier_indices, opt$`output-dir`)
  
  # Print summary
  cat("\n============================================================\n")
  cat("Outlier Detection Summary\n")
  cat("============================================================\n")
  cat(sprintf("Method: %s\n", opt$method))
  cat(sprintf("Distance metric: %s\n", opt$`distance-metric`))
  cat(sprintf("Total outliers: %d\n", length(outlier_indices)))
  cat(sprintf("Outlier video: %s\n", ifelse(is.null(video_path), "Failed", basename(video_path))))
  cat(sprintf("\nAll outputs saved to: %s\n", opt$`output-dir`))
}

# Run main function
if (!interactive()) {
  main()
}

