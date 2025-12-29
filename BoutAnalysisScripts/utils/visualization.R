# Visualization Utilities
# Functions for creating plots and visualizations

library(ggplot2)
library(gridExtra)
library(factoextra)

#' Create PCA biplot
#'
#' @param pca_result Result from prcomp()
#' @param clusters Optional cluster assignments
#' @param title Plot title
#' @return ggplot object
plot_pca_biplot <- function(pca_result, clusters = NULL, title = "PCA Biplot") {
  if (!is.null(clusters)) {
    p <- fviz_pca_biplot(pca_result, 
                        col.ind = as.factor(clusters),
                        palette = "Set2",
                        addEllipses = TRUE,
                        title = title)
  } else {
    p <- fviz_pca_biplot(pca_result, title = title)
  }
  return(p)
}

#' Plot cluster assignments in PCA space
#'
#' @param pca_result Result from prcomp()
#' @param clusters Cluster assignments
#' @param method_name Name of clustering method
#' @return ggplot object
plot_cluster_pca <- function(pca_result, clusters, method_name = "Clustering") {
  p <- fviz_cluster(list(data = pca_result$x, cluster = clusters),
                   geom = "point",
                   main = sprintf("%s in PCA Space", method_name),
                   palette = "Set2")
  return(p)
}

#' Plot dendrogram for hierarchical clustering
#'
#' @param hc_result Result from hclust()
#' @param k Number of clusters to highlight
#' @param title Plot title
#' @return ggplot object
plot_dendrogram <- function(hc_result, k = NULL, title = "Hierarchical Clustering Dendrogram") {
  if (!is.null(k)) {
    p <- fviz_dend(hc_result, k = k, 
                  cex = 0.5,
                  k_colors = "Set2",
                  main = title)
  } else {
    p <- fviz_dend(hc_result,
                  cex = 0.5,
                  main = title)
  }
  return(p)
}

#' Plot outlier score distributions
#'
#' @param scores Named list of outlier score vectors
#' @param title Plot title
#' @return ggplot object
plot_outlier_scores <- function(scores, title = "Outlier Score Distributions") {
  # Prepare data
  score_df <- data.frame(
    Method = rep(names(scores), sapply(scores, length)),
    Score = unlist(scores)
  )
  
  p <- ggplot(score_df, aes(x = Score, fill = Method)) +
    geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
    facet_wrap(~ Method, scales = "free") +
    labs(title = title, x = "Outlier Score", y = "Frequency") +
    theme_minimal() +
    theme(legend.position = "none")
  
  return(p)
}

#' Plot outlier rankings
#'
#' @param outlier_df Data frame with outlier information
#' @param score_col Column name for outlier score
#' @param n_top Number of top outliers to show
#' @param title Plot title
#' @return ggplot object
plot_outlier_rankings <- function(outlier_df, score_col, n_top = 20, title = "Top Outliers") {
  # Sort by score
  outlier_df_sorted <- outlier_df[order(outlier_df[[score_col]], decreasing = TRUE), ]
  top_outliers <- head(outlier_df_sorted, n_top)
  
  # Create labels
  top_outliers$label <- sprintf("%s (ID: %d)", 
                                top_outliers$video_name, 
                                top_outliers$bout_id)
  
  p <- ggplot(top_outliers, aes(x = reorder(label, get(score_col)), y = get(score_col))) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = title, x = "Bout", y = "Outlier Score") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  
  return(p)
}

#' Plot feature contribution heatmap
#'
#' @param contributions_df Data frame with feature contributions
#' @param n_top_features Number of top features to show
#' @param n_top_outliers Number of top outliers to show
#' @param title Plot title
#' @return ggplot object
plot_feature_contributions <- function(contributions_df, n_top_features = 10, 
                                       n_top_outliers = 20, title = "Feature Contributions") {
  # Get top outliers
  top_outliers <- unique(contributions_df$bout_id)[1:min(n_top_outliers, length(unique(contributions_df$bout_id)))]
  
  # Get top features by average absolute contribution
  feature_importance <- contributions_df %>%
    filter(bout_id %in% top_outliers) %>%
    group_by(feature_name) %>%
    summarize(avg_abs_contrib = mean(abs(z_score), na.rm = TRUE)) %>%
    arrange(desc(avg_abs_contrib)) %>%
    head(n_top_features)
  
  # Filter data
  plot_data <- contributions_df %>%
    filter(bout_id %in% top_outliers,
           feature_name %in% feature_importance$feature_name) %>%
    mutate(bout_label = sprintf("Bout %d", bout_id))
  
  p <- ggplot(plot_data, aes(x = feature_name, y = bout_label, fill = z_score)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                         midpoint = 0, na.value = "grey") +
    labs(title = title, x = "Feature", y = "Outlier Bout", fill = "Z-Score") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
          axis.text.y = element_text(size = 8))
  
  return(p)
}

#' Plot outlier feature comparison
#'
#' @param outlier_df Data frame with outlier information
#' @param feature_cols Feature column names
#' @param n_outliers Number of outliers to compare
#' @param title Plot title
#' @return ggplot object
plot_outlier_feature_comparison <- function(outlier_df, feature_cols, n_outliers = 5, title = "Outlier Feature Comparison") {
  # Select top outliers
  top_outliers <- head(outlier_df, n_outliers)
  
  # Prepare data for plotting
  plot_data_list <- list()
  for (i in seq_len(nrow(top_outliers))) {
    bout <- top_outliers[i, ]
    for (feat in feature_cols) {
      if (feat %in% names(bout)) {
        plot_data_list[[length(plot_data_list) + 1]] <- list(
          bout_id = bout$bout_id,
          video_name = bout$video_name,
          feature = feat,
          value = bout[[feat]]
        )
      }
    }
  }
  
  plot_data <- do.call(rbind, lapply(plot_data_list, function(x) data.frame(x, stringsAsFactors = FALSE)))
  
  p <- ggplot(plot_data, aes(x = feature, y = value, color = as.factor(bout_id))) +
    geom_point(size = 3) +
    labs(title = title, x = "Feature", y = "Value", color = "Bout ID") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(p)
}

#' Create summary plot of outlier analysis
#'
#' @param explanations_df Data frame with outlier explanations
#' @param output_file Path to save plot
create_outlier_summary_plot <- function(explanations_df, output_file) {
  # Create multiple panels
  p1 <- plot_outlier_scores(list(
    Mahalanobis = explanations_df$outlier_score_mahalanobis,
    LOF = explanations_df$outlier_score_lof,
    Isolation = explanations_df$outlier_score_isolation
  ), title = "Outlier Score Distributions")
  
  p2 <- plot_outlier_rankings(explanations_df, "outlier_score_mahalanobis", 
                              n_top = 15, title = "Top Mahalanobis Outliers")
  
  # Combine plots
  combined <- grid.arrange(p1, p2, ncol = 1, heights = c(1, 1.5))
  
  ggsave(output_file, combined, width = 12, height = 10, dpi = 300)
  cat(sprintf("Summary plot saved to: %s\n", output_file))
}

