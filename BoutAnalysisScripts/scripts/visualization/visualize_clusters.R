#!/usr/bin/env Rscript
# Create comprehensive visualizations of behavior bout clusters.
#
# This script generates:
# - Dimensionality reduction plots (PCA, t-SNE)
# - Feature distribution plots
# - Cluster analysis plots
# - Bout-level visualizations

# Add default library paths (user library first)
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(ggplot2)
  library(Rtsne)
  library(factoextra)
  library(pheatmap)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-f", "--features"), type="character", default="bout_features.csv",
              help="Input CSV file with bout features (default: bout_features.csv)"),
  make_option(c("-c", "--clusters"), type="character", default="cluster_assignments_kmeans.csv",
              help="Input CSV file with cluster assignments (default: cluster_assignments_kmeans.csv)"),
  make_option(c("-o", "--output-dir"), type="character", default=".",
              help="Output directory for plots (default: current directory)"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Separate metadata and features
separate_metadata_and_features <- function(df) {
  metadata_cols <- c("bout_id", "video_name", "animal_id", "start_frame", 
                     "end_frame", "behavior", "duration_frames", "cluster_id")
  
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
  
  X_scaled <- scale(X)
  X_scaled[!is.finite(X_scaled)] <- 0
  
  return(list(metadata=metadata_df, features=X_scaled, feature_names=feature_names_filtered))
}

# Plot PCA reduction
plot_pca_reduction <- function(X, labels, output_path, metadata_df = NULL) {
  pca_result <- prcomp(X, scale.=FALSE)
  
  pca_df <- data.frame(
    PC1 = pca_result$x[, 1],
    PC2 = pca_result$x[, 2],
    cluster = as.factor(labels)
  )
  
  if (!is.null(metadata_df) && "animal_id" %in% colnames(metadata_df)) {
    pca_df$animal_id <- metadata_df$animal_id
  }
  
  # Plot 1: Colored by cluster
  p1 <- ggplot(pca_df, aes(x=PC1, y=PC2, color=cluster)) +
    geom_point(alpha=0.6, size=3) +
    labs(
      x = sprintf("PC1 (%.1f%% variance)", summary(pca_result)$importance[2, 1] * 100),
      y = sprintf("PC2 (%.1f%% variance)", summary(pca_result)$importance[2, 2] * 100),
      title = "PCA: Clusters",
      color = "Cluster ID",
      caption = "Interpretation: PCA reduces high-dimensional features to 2D. Each point is a bout colored by cluster. Points close together have similar features. Clusters should form distinct groups if clustering is meaningful."
    ) +
    theme_minimal() +
    theme(legend.position="right",
          plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
  
  # Plot 2: Colored by animal if available
  if ("animal_id" %in% colnames(pca_df)) {
    p2 <- ggplot(pca_df, aes(x=PC1, y=PC2, color=animal_id)) +
      geom_point(alpha=0.6, size=3) +
      labs(
        x = sprintf("PC1 (%.1f%% variance)", summary(pca_result)$importance[2, 1] * 100),
        y = sprintf("PC2 (%.1f%% variance)", summary(pca_result)$importance[2, 2] * 100),
        title = "PCA: Animals",
        color = "Animal ID",
        caption = "Interpretation: Same PCA space colored by animal ID. Clustering by animal suggests individual differences in behavior patterns."
      ) +
      theme_minimal() +
      theme(legend.position="right",
            plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
    
    p <- gridExtra::grid.arrange(p1, p2, ncol=2)
  } else {
    p <- p1
  }
  
  ggsave(output_path, p, width=16, height=6, dpi=300)
  cat(sprintf("Saved PCA plot to %s\n", output_path))
}

# Plot t-SNE reduction
plot_tsne_reduction <- function(X, labels, output_path, metadata_df = NULL) {
  if (nrow(X) < 5) {
    cat(sprintf("Too few samples (%d) for t-SNE. Skipping.\n", nrow(X)))
    return
  }
  
  # Adjust perplexity based on sample size
  n_samples <- nrow(X)
  perplexity <- min(30, max(5, floor((n_samples - 1) / 3)))
  
  tsne_result <- Rtsne(X, dims=2, perplexity=perplexity, verbose=FALSE)
  
  tsne_df <- data.frame(
    tSNE1 = tsne_result$Y[, 1],
    tSNE2 = tsne_result$Y[, 2],
    cluster = as.factor(labels)
  )
  
  if (!is.null(metadata_df) && "animal_id" %in% colnames(metadata_df)) {
    tsne_df$animal_id <- metadata_df$animal_id
  }
  
  # Plot 1: Colored by cluster
  p1 <- ggplot(tsne_df, aes(x=tSNE1, y=tSNE2, color=cluster)) +
    geom_point(alpha=0.6, size=3) +
    labs(
      x = "t-SNE Dimension 1",
      y = "t-SNE Dimension 2",
      title = "t-SNE: Clusters",
      color = "Cluster ID",
      caption = "Interpretation: t-SNE is a non-linear dimensionality reduction that preserves local neighborhoods. Points close in original space remain close in 2D. Clusters should appear as distinct groups. Unlike PCA, t-SNE distances are not directly interpretable."
    ) +
    theme_minimal() +
    theme(legend.position="right",
          plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
  
  # Plot 2: Colored by animal if available
  if ("animal_id" %in% colnames(tsne_df)) {
    p2 <- ggplot(tsne_df, aes(x=tSNE1, y=tSNE2, color=animal_id)) +
      geom_point(alpha=0.6, size=3) +
      labs(
        x = "t-SNE Dimension 1",
        y = "t-SNE Dimension 2",
        title = "t-SNE: Animals",
        color = "Animal ID",
        caption = "Interpretation: Same t-SNE space colored by animal ID. Clustering by animal suggests individual differences in behavior patterns."
      ) +
      theme_minimal() +
      theme(legend.position="right",
            plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
    
    p <- gridExtra::grid.arrange(p1, p2, ncol=2)
  } else {
    p <- p1
  }
  
  ggsave(output_path, p, width=16, height=6, dpi=300)
  cat(sprintf("Saved t-SNE plot to %s\n", output_path))
}

# Plot feature distributions
plot_feature_distributions <- function(df, labels, output_path, n_features = 12) {
  # Get feature columns
  metadata_cols <- c("bout_id", "video_name", "animal_id", "start_frame", 
                     "end_frame", "behavior", "duration_frames", "cluster_id")
  feature_cols <- setdiff(colnames(df), metadata_cols)
  
  if (length(feature_cols) == 0) {
    cat("No feature columns found\n")
    return
  }
  
  # Select top features by variance
  feature_vars <- apply(df[, feature_cols, drop=FALSE], 2, var, na.rm=TRUE)
  top_features <- names(sort(feature_vars, decreasing=TRUE))[1:min(n_features, length(feature_cols))]
  
  # Prepare data for plotting
  plot_data <- data.frame(
    cluster = as.factor(labels)
  )
  
  for (feature in top_features) {
    plot_data[[feature]] <- df[[feature]]
  }
  
  # Create violin plots
  plot_list <- list()
  for (feature in top_features) {
    p <- ggplot(plot_data, aes_string(x="cluster", y=feature, fill="cluster")) +
      geom_violin(alpha=0.7) +
      geom_boxplot(width=0.1, alpha=0.5) +
      labs(title=feature, x="Cluster ID", y="Feature Value",
           caption="Interpretation: Violin plots show the distribution shape; boxplots show quartiles. Differences between clusters indicate distinguishing features.") +
      theme_minimal() +
      theme(legend.position="none",
            plot.caption = element_text(hjust=0, size=7, margin=margin(t=8)))
    
    plot_list[[feature]] <- p
  }
  
  # Arrange plots
  n_cols <- 3
  n_rows <- ceiling(length(plot_list) / n_cols)
  p <- gridExtra::grid.arrange(grobs=plot_list, ncol=n_cols, nrow=n_rows)
  
  ggsave(output_path, p, width=18, height=6*n_rows, dpi=300)
  cat(sprintf("Saved feature distribution plot to %s\n", output_path))
}

# Plot cluster heatmap
plot_cluster_heatmap <- function(X, labels, feature_names, output_path) {
  unique_labels <- sort(unique(labels))
  unique_labels <- unique_labels[unique_labels != 0]  # Remove noise
  
  # Check if we have valid clusters
  if (length(unique_labels) == 0 || ncol(X) == 0) {
    cat("Warning: No valid clusters for heatmap (only noise or empty data)\n")
    # Create a simple placeholder plot
    png(output_path, width=12, height=6, units="in", res=300)
    plot.new()
    text(0.5, 0.5, "No valid clusters for heatmap visualization", cex=1.2)
    dev.off()
    return()
  }
  
  # Calculate mean features per cluster
  cluster_means <- matrix(0, nrow=length(unique_labels), ncol=ncol(X))
  if (length(unique_labels) > 0) {
    rownames(cluster_means) <- paste0("Cluster ", unique_labels)
  }
  
  for (i in seq_along(unique_labels)) {
    cluster_id <- unique_labels[i]
    cluster_mask <- labels == cluster_id
    if (sum(cluster_mask) > 0) {
      cluster_means[i, ] <- colMeans(X[cluster_mask, , drop=FALSE], na.rm=TRUE)
    }
  }
  
  # Select top features by variance across clusters
  if (nrow(cluster_means) > 1 && ncol(cluster_means) > 0) {
    feature_vars <- apply(cluster_means, 2, var, na.rm=TRUE)
    top_indices <- order(feature_vars, decreasing=TRUE)[1:min(20, length(feature_vars))]
    
    heatmap_data <- cluster_means[, top_indices, drop=FALSE]
    if (length(top_indices) > 0 && length(feature_names) >= max(top_indices)) {
      colnames(heatmap_data) <- feature_names[top_indices]
    }
    
    png(output_path, width=12, height=max(7, length(unique_labels)*0.5 + 1), units="in", res=300)
    par(mar=c(4, 4, 4, 2) + 0.1)
    pheatmap(heatmap_data,
             cluster_rows=FALSE,
             cluster_cols=TRUE,
             annotation_legend=TRUE,
             main="Feature Means by Cluster (Top 20 Features)",
             fontsize=8,
             angle_col=45)
    # Add caption below heatmap using grid graphics
    grid::grid.text("Interpretation: Heatmap shows mean feature values per cluster. Colors: red=high, blue=low, white=average. Similar color patterns indicate similar feature profiles.", 
                    x=0.5, y=0.02, just=c("center", "bottom"), gp=grid::gpar(fontsize=8, col="black"))
    dev.off()
    
    cat(sprintf("Saved cluster heatmap to %s\n", output_path))
  } else {
    cat("Warning: Insufficient clusters or features for heatmap\n")
    png(output_path, width=12, height=6, units="in", res=300)
    plot.new()
    text(0.5, 0.5, "Insufficient data for heatmap visualization", cex=1.2)
    dev.off()
  }
}

# Plot cluster sizes
plot_cluster_sizes <- function(labels, output_path) {
  label_counts <- table(labels)
  label_df <- data.frame(
    cluster_id = names(label_counts),
    count = as.numeric(label_counts)
  )
  
  p <- ggplot(label_df, aes(x=cluster_id, y=count, fill=cluster_id)) +
    geom_bar(stat="identity", alpha=0.7) +
    geom_text(aes(label=count), vjust=-0.5) +
    labs(
      x = "Cluster ID",
      y = "Number of Bouts",
      title = "Cluster Size Distribution"
    ) +
    theme_minimal() +
    theme(legend.position="none")
  
  ggsave(output_path, p, width=10, height=6, dpi=300)
  cat(sprintf("Saved cluster size plot to %s\n", output_path))
}

# Plot bout timeline
plot_bout_timeline <- function(results_df, output_path) {
  # Group by video
  results_df$video_name <- factor(results_df$video_name)
  results_df$cluster_id <- factor(results_df$cluster_id)
  
  p <- ggplot(results_df, aes(x=start_frame, xend=end_frame, y=video_name, yend=video_name, 
                              color=cluster_id)) +
    geom_segment(size=2, alpha=0.7) +
    labs(
      x = "Frame Number",
      y = "Video",
      title = "Bout Timeline Colored by Cluster",
      color = "Cluster ID",
      caption = "Interpretation: Each horizontal line represents a video. Colored segments show bout locations and durations, colored by cluster assignment. This reveals temporal patterns: whether certain clusters occur at specific times, cluster co-occurrence, and distribution across videos."
    ) +
    theme_minimal() +
    theme(axis.text.y=element_text(size=6),
          plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
  
  ggsave(output_path, p, width=16, height=max(8, length(unique(results_df$video_name))*0.5), 
         dpi=300, limitsize=FALSE)
  cat(sprintf("Saved bout timeline to %s\n", output_path))
}

# Main execution
main <- function() {
  cat(sprintf("Loading features from %s\n", opt$features))
  features_df <- read.csv(opt$features, stringsAsFactors=FALSE)
  
  cat(sprintf("Loading cluster assignments from %s\n", opt$clusters))
  clusters_df <- read.csv(opt$clusters, stringsAsFactors=FALSE)
  
  # Merge data
  df <- merge(features_df, clusters_df[, c("bout_id", "cluster_id")], 
             by="bout_id", all.x=FALSE)
  
  # Prepare features
  prepared <- prepare_features(df)
  metadata_df <- prepared$metadata
  X_scaled <- prepared$features
  feature_names <- prepared$feature_names
  
  # Check if we have valid data
  if (nrow(X_scaled) == 0 || ncol(X_scaled) == 0) {
    stop("No valid features after preprocessing. Check input files.")
  }
  
  # Get cluster labels (must match X_scaled rows)
  labels <- clusters_df$cluster_id[match(df$bout_id, clusters_df$bout_id)]
  
  cat("Creating visualizations...\n")
  
  # PCA plot
  plot_pca_reduction(X_scaled, labels, 
                    file.path(opt$`output-dir`, "pca_clusters.png"),
                    metadata_df)
  
  # t-SNE plot
  plot_tsne_reduction(X_scaled, labels,
                     file.path(opt$`output-dir`, "tsne_clusters.png"),
                     metadata_df)
  
  # Feature distributions
  plot_feature_distributions(df, labels,
                            file.path(opt$`output-dir`, "feature_distributions.png"))
  
  # Cluster heatmap
  plot_cluster_heatmap(X_scaled, labels, feature_names,
                      file.path(opt$`output-dir`, "cluster_heatmap.png"))
  
  # Cluster sizes
  plot_cluster_sizes(labels, file.path(opt$`output-dir`, "cluster_sizes.png"))
  
  # Bout timeline
  plot_bout_timeline(clusters_df, file.path(opt$`output-dir`, "bout_timeline.png"))
  
  cat("Visualization complete!\n")
}

# Run main function
if (!interactive()) {
  main()
}

