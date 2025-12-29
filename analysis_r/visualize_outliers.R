#!/usr/bin/env Rscript
# Generate visualizations for outlier detection analysis
#
# This script creates comprehensive plots for outlier results:
# - Outlier score distributions
# - Outlier rankings
# - PCA plots with outliers highlighted
# - Feature contribution heatmaps
# - Outlier comparison plots
#
# Usage:
#   Rscript analysis_r/visualize_outliers.R --features bout_features.csv --output-dir results/outlier_detection

# Add default library paths
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  library(factoextra)
  library(pheatmap)
  library(tidyr)
  library(tibble)
  library(stringr)
})

# Source utility functions
source("analysis_r/utils/data_preprocessing.R")

# Parse command line arguments
option_list <- list(
  make_option(c("-f", "--features"), type="character", default="results/bout_features.csv",
              help="Input CSV file with bout features (default: results/bout_features.csv)"),
  make_option(c("-o", "--output-dir"), type="character", default="results/outlier_detection",
              help="Output directory for visualizations (default: results/outlier_detection)"),
  make_option(c("-e", "--explanations"), type="character", default="results/outlier_detection/outlier_explanations.csv",
              help="Outlier explanations CSV file (default: results/outlier_detection/outlier_explanations.csv)"),
  make_option(c("-c", "--contributions"), type="character", default="results/outlier_detection/outlier_feature_contributions.csv",
              help="Feature contributions CSV file (default: results/outlier_detection/outlier_feature_contributions.csv)"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Main execution
main <- function() {
  cat("============================================================\n")
  cat("Outlier Detection Visualization\n")
  cat("============================================================\n\n")
  
  # Create output directory
  dir.create(opt$`output-dir`, showWarnings = FALSE, recursive = TRUE)
  
  # Close any open graphics devices to prevent Rplots.pdf in main directory
  while (dev.cur() != 1) {
    dev.off()
  }
  
  # Suppress default PDF device (prevents Rplots.pdf from being created)
  pdf(NULL)
  
  # Load outlier explanations
  cat(sprintf("Loading outlier explanations from: %s\n", opt$explanations))
  if (!file.exists(opt$explanations)) {
    stop(sprintf("Outlier explanations file not found: %s", opt$explanations))
  }
  
  explanations_df <- read.csv(opt$explanations, stringsAsFactors = FALSE)
  cat(sprintf("Loaded %d outlier records\n", nrow(explanations_df)))
  
  # Load features for PCA
  cat(sprintf("Loading features from: %s\n", opt$features))
  if (!file.exists(opt$features)) {
    stop(sprintf("Features file not found: %s", opt$features))
  }
  
  df <- read.csv(opt$features, stringsAsFactors = FALSE)
  
  # Prepare features
  exclude_cols <- c("bout_id", "video_name", "animal_id", "start_frame", "end_frame", "behavior")
  prep_result <- prepare_features(df, exclude_cols = exclude_cols)
  processed_df <- prep_result$processed_df
  feature_cols <- prep_result$feature_cols
  
  feature_matrix <- as.matrix(processed_df[, feature_cols, drop = FALSE])
  feature_matrix[!is.finite(feature_matrix)] <- 0
  
  # Load PCA results (but recompute if dimensions don't match)
  pca_file <- file.path(dirname(opt$`output-dir`), "pca_results.RData")
  pca_result <- NULL
  if (file.exists(pca_file)) {
    load(pca_file)
    cat("Loaded PCA results\n")
    # pca_result is a list with pca_result$pca_result being the actual prcomp object
    if (is.list(pca_result) && "pca_result" %in% names(pca_result)) {
      pca_result <- pca_result$pca_result
    }
    # Check if PCA dimensions match current data
    if (!is.null(pca_result) && nrow(pca_result$x) != nrow(feature_matrix)) {
      cat(sprintf("PCA results have %d rows but data has %d rows. Recomputing PCA on full dataset...\n", 
                  nrow(pca_result$x), nrow(feature_matrix)))
      pca_result <- NULL
    }
  }
  
  if (is.null(pca_result)) {
    cat("Computing PCA on full dataset...\n")
    pca_result <- prcomp(feature_matrix, center = TRUE, scale. = TRUE)
  }
  
  cat("\nGenerating visualizations...\n")
  
  # Merge explanations with original data to get video_name
  explanations_df <- explanations_df %>%
    left_join(df[, c("bout_id", "video_name", "animal_id")], by = "bout_id")
  
  # Fill in any missing video_name from explanations_df itself
  if (!"video_name" %in% names(explanations_df) || any(is.na(explanations_df$video_name))) {
    # video_name should already be in explanations_df from the find_outliers.R script
    if (!"video_name" %in% names(explanations_df)) {
      explanations_df$video_name <- df$video_name[match(explanations_df$bout_id, df$bout_id)]
    }
  }
  
  # 1. Outlier Score Distributions
  cat("  [1/10] Outlier Score Distributions\n")
  score_df <- data.frame(
    Method = rep(c("Mahalanobis", "LOF", "Isolation Forest"), each = nrow(explanations_df)),
    Score = c(explanations_df$outlier_score_mahalanobis,
              explanations_df$outlier_score_lof,
              explanations_df$outlier_score_isolation)
  )
  
  p <- ggplot(score_df, aes(x = Score, fill = Method)) +
    geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
    facet_wrap(~ Method, scales = "free") +
    labs(title = "Outlier Score Distributions",
         x = "Outlier Score",
         y = "Frequency") +
    theme_minimal() +
    theme(legend.position = "none")
  
  ggsave(file.path(opt$`output-dir`, "outlier_score_distributions.png"), p, 
         width = 12, height = 4, dpi = 300)
  
  # 2. Top Outliers by Method
  cat("  [2/10] Top Outliers - Mahalanobis\n")
  top_mahal <- explanations_df %>%
    arrange(desc(outlier_score_mahalanobis)) %>%
    head(20) %>%
    mutate(label = sprintf("%s (ID: %d)", video_name, bout_id))
  
  p <- ggplot(top_mahal, aes(x = reorder(label, outlier_score_mahalanobis), 
                             y = outlier_score_mahalanobis)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Top 20 Outliers - Mahalanobis Distance",
         x = "Bout",
         y = "Mahalanobis Distance") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 7))
  
  ggsave(file.path(opt$`output-dir`, "top_outliers_mahalanobis.png"), p, 
         width = 10, height = 8, dpi = 300)
  
  cat("  [3/10] Top Outliers - LOF\n")
  top_lof <- explanations_df %>%
    arrange(desc(outlier_score_lof)) %>%
    head(20) %>%
    mutate(label = sprintf("%s (ID: %d)", video_name, bout_id))
  
  p <- ggplot(top_lof, aes(x = reorder(label, outlier_score_lof), y = outlier_score_lof)) +
    geom_bar(stat = "identity", fill = "darkgreen") +
    coord_flip() +
    labs(title = "Top 20 Outliers - Local Outlier Factor",
         x = "Bout",
         y = "LOF Score") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 7))
  
  ggsave(file.path(opt$`output-dir`, "top_outliers_lof.png"), p, 
         width = 10, height = 8, dpi = 300)
  
  cat("  [4/10] Top Outliers - Isolation Forest\n")
  top_iso <- explanations_df %>%
    arrange(desc(outlier_score_isolation)) %>%
    head(20) %>%
    mutate(label = sprintf("%s (ID: %d)", video_name, bout_id))
  
  p <- ggplot(top_iso, aes(x = reorder(label, outlier_score_isolation), 
                           y = outlier_score_isolation)) +
    geom_bar(stat = "identity", fill = "orange") +
    coord_flip() +
    labs(title = "Top 20 Outliers - Isolation Forest",
         x = "Bout",
         y = "Isolation Forest Score") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 7))
  
  ggsave(file.path(opt$`output-dir`, "top_outliers_isolation.png"), p, 
         width = 10, height = 8, dpi = 300)
  
  # 5. Outliers in PCA Space
  cat("  [5/10] Outliers in PCA Space\n")
  pca_df <- data.frame(
    PC1 = pca_result$x[, 1],
    PC2 = pca_result$x[, 2],
    bout_id = processed_df$bout_id,
    is_mahal_outlier = explanations_df$is_outlier_mahalanobis[match(processed_df$bout_id, explanations_df$bout_id)],
    is_lof_outlier = explanations_df$is_outlier_lof[match(processed_df$bout_id, explanations_df$bout_id)],
    is_iso_outlier = explanations_df$is_outlier_isolation[match(processed_df$bout_id, explanations_df$bout_id)],
    is_consensus = explanations_df$consensus_outlier[match(processed_df$bout_id, explanations_df$bout_id)]
  )
  pca_df$is_mahal_outlier[is.na(pca_df$is_mahal_outlier)] <- FALSE
  pca_df$is_lof_outlier[is.na(pca_df$is_lof_outlier)] <- FALSE
  pca_df$is_iso_outlier[is.na(pca_df$is_iso_outlier)] <- FALSE
  pca_df$is_consensus[is.na(pca_df$is_consensus)] <- FALSE
  
  p1 <- ggplot(pca_df, aes(x = PC1, y = PC2)) +
    geom_point(aes(color = is_mahal_outlier), alpha = 0.6, size = 0.8) +
    scale_color_manual(values = c("FALSE" = "gray", "TRUE" = "red"), 
                      labels = c("Normal", "Outlier")) +
    labs(title = "Mahalanobis Outliers in PCA Space",
         x = "Principal Component 1",
         y = "Principal Component 2") +
    theme_minimal()
  
  p2 <- ggplot(pca_df, aes(x = PC1, y = PC2)) +
    geom_point(aes(color = is_lof_outlier), alpha = 0.6, size = 0.8) +
    scale_color_manual(values = c("FALSE" = "gray", "TRUE" = "red"),
                      labels = c("Normal", "Outlier")) +
    labs(title = "LOF Outliers in PCA Space",
         x = "Principal Component 1",
         y = "Principal Component 2") +
    theme_minimal()
  
  p3 <- ggplot(pca_df, aes(x = PC1, y = PC2)) +
    geom_point(aes(color = is_consensus), alpha = 0.6, size = 0.8) +
    scale_color_manual(values = c("FALSE" = "gray", "TRUE" = "red"),
                      labels = c("Normal", "Consensus Outlier")) +
    labs(title = "Consensus Outliers in PCA Space",
         x = "Principal Component 1",
         y = "Principal Component 2") +
    theme_minimal()
  
  combined <- grid.arrange(p1, p2, p3, ncol = 3)
  ggsave(file.path(opt$`output-dir`, "outliers_pca_space.png"), combined, 
         width = 18, height = 6, dpi = 300)
  
  # 6. Outlier Method Comparison
  cat("  [6/10] Outlier Method Comparison\n")
  method_comparison <- data.frame(
    Method = c("Mahalanobis", "LOF", "Isolation Forest", "Consensus"),
    Count = c(sum(explanations_df$is_outlier_mahalanobis, na.rm = TRUE),
              sum(explanations_df$is_outlier_lof, na.rm = TRUE),
              sum(explanations_df$is_outlier_isolation, na.rm = TRUE),
              sum(explanations_df$consensus_outlier, na.rm = TRUE))
  )
  
  p <- ggplot(method_comparison, aes(x = Method, y = Count, fill = Method)) +
    geom_bar(stat = "identity") +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Number of Outliers Identified by Each Method",
         x = "Method",
         y = "Number of Outliers") +
    theme_minimal() +
    theme(legend.position = "none")
  
  ggsave(file.path(opt$`output-dir`, "outlier_method_comparison.png"), p, 
         width = 8, height = 6, dpi = 300)
  
  # 7. Score Correlation
  cat("  [7/10] Outlier Score Correlations\n")
  score_cor <- cor(explanations_df[, c("outlier_score_mahalanobis", 
                                       "outlier_score_lof", 
                                       "outlier_score_isolation")], 
                   use = "complete.obs")
  
  png(file.path(opt$`output-dir`, "outlier_score_correlation.png"), 
      width = 1000, height = 1000, res = 300)
  pheatmap(score_cor, 
           display_numbers = TRUE,
           number_format = "%.2f",
           main = "Correlation Between Outlier Scores",
           color = colorRampPalette(c("blue", "white", "red"))(100))
  dev.off()
  
  # 8. Feature Contributions (if available)
  if (file.exists(opt$contributions)) {
    cat("  [8/10] Feature Contributions Heatmap\n")
    contributions_df <- read.csv(opt$contributions, stringsAsFactors = FALSE)
    
    # Get top outliers and top features
    top_outlier_ids <- explanations_df %>%
      arrange(desc(outlier_score_mahalanobis)) %>%
      head(15) %>%
      pull(bout_id)
    
    top_features <- contributions_df %>%
      filter(bout_id %in% top_outlier_ids) %>%
      group_by(feature_name) %>%
      summarize(avg_abs_z = mean(abs(z_score), na.rm = TRUE)) %>%
      arrange(desc(avg_abs_z)) %>%
      head(20) %>%
      pull(feature_name)
    
    heatmap_data <- contributions_df %>%
      filter(bout_id %in% top_outlier_ids, feature_name %in% top_features) %>%
      select(bout_id, feature_name, z_score) %>%
      tidyr::pivot_wider(names_from = feature_name, values_from = z_score, values_fill = 0) %>%
      column_to_rownames("bout_id") %>%
      as.matrix()
    
    png(file.path(opt$`output-dir`, "feature_contributions_heatmap.png"), 
        width = 2000, height = 1200, res = 300)
    pheatmap(heatmap_data,
             main = "Feature Contributions for Top Outliers",
             color = colorRampPalette(c("blue", "white", "red"))(100),
             cluster_rows = TRUE,
             cluster_cols = TRUE,
             fontsize = 8)
    dev.off()
  }
  
  # 9. Outlier Score Box Plots
  cat("  [9/10] Outlier Score Box Plots\n")
  score_long <- explanations_df %>%
    select(outlier_score_mahalanobis, outlier_score_lof, outlier_score_isolation) %>%
    tidyr::pivot_longer(everything(), names_to = "Method", values_to = "Score") %>%
    mutate(Method = gsub("outlier_score_", "", Method),
           Method = gsub("_", " ", Method),
           Method = stringr::str_to_title(Method))
  
  p <- ggplot(score_long, aes(x = Method, y = Score, fill = Method)) +
    geom_boxplot(alpha = 0.7) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Distribution of Outlier Scores by Method",
         x = "Method",
         y = "Outlier Score") +
    theme_minimal() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave(file.path(opt$`output-dir`, "outlier_score_boxplots.png"), p, 
         width = 10, height = 6, dpi = 300)
  
  # 10. Summary Statistics Plot
  cat("  [10/10] Summary Statistics\n")
  summary_stats <- explanations_df %>%
    summarize(
      Total_Bouts = n(),
      Mahalanobis_Outliers = sum(is_outlier_mahalanobis, na.rm = TRUE),
      LOF_Outliers = sum(is_outlier_lof, na.rm = TRUE),
      Isolation_Outliers = sum(is_outlier_isolation, na.rm = TRUE),
      Consensus_Outliers = sum(consensus_outlier, na.rm = TRUE),
      Mean_Mahal_Score = mean(outlier_score_mahalanobis, na.rm = TRUE),
      Mean_LOF_Score = mean(outlier_score_lof, na.rm = TRUE),
      Mean_ISO_Score = mean(outlier_score_isolation, na.rm = TRUE)
    )
  
  # Create a text summary plot
  png(file.path(opt$`output-dir`, "outlier_summary.png"), width = 1200, height = 800, res = 300)
  par(mar = c(0, 0, 0, 0))
  plot.new()
  text(0.5, 0.5, 
       sprintf("Outlier Detection Summary\n\nTotal Bouts: %d\n\nMahalanobis Outliers: %d (%.1f%%)\nLOF Outliers: %d (%.1f%%)\nIsolation Forest Outliers: %d (%.1f%%)\nConsensus Outliers: %d (%.1f%%)\n\nMean Scores:\nMahalanobis: %.2f\nLOF: %.2f\nIsolation Forest: %.2f",
               summary_stats$Total_Bouts,
               summary_stats$Mahalanobis_Outliers, 100*summary_stats$Mahalanobis_Outliers/summary_stats$Total_Bouts,
               summary_stats$LOF_Outliers, 100*summary_stats$LOF_Outliers/summary_stats$Total_Bouts,
               summary_stats$Isolation_Outliers, 100*summary_stats$Isolation_Outliers/summary_stats$Total_Bouts,
               summary_stats$Consensus_Outliers, 100*summary_stats$Consensus_Outliers/summary_stats$Total_Bouts,
               summary_stats$Mean_Mahal_Score,
               summary_stats$Mean_LOF_Score,
               summary_stats$Mean_ISO_Score),
       cex = 1.5, font = 2)
  dev.off()
  
  cat("\n============================================================\n")
  cat("Outlier detection visualizations complete!\n")
  cat(sprintf("Plots saved to: %s\n", opt$`output-dir`))
  cat("\nGenerated plots:\n")
  cat("  - outlier_score_distributions.png\n")
  cat("  - top_outliers_mahalanobis.png\n")
  cat("  - top_outliers_lof.png\n")
  cat("  - top_outliers_isolation.png\n")
  cat("  - outliers_pca_space.png\n")
  cat("  - outlier_method_comparison.png\n")
  cat("  - outlier_score_correlation.png\n")
  cat("  - feature_contributions_heatmap.png\n")
  cat("  - outlier_score_boxplots.png\n")
  cat("  - outlier_summary.png\n")
  
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
  # Try to move to outlier_detection directory (default)
  output_dir <- "results/outlier_detection"
  if (dir.exists(output_dir)) {
    file.rename("Rplots.pdf", file.path(output_dir, "Rplots.pdf"))
  }
}

