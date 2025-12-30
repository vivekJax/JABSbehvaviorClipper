#!/usr/bin/env Rscript
# Fix R environment issues - set a valid editor or remove the option
tryCatch({
  if (system("which nano > /dev/null 2>&1", ignore.stdout = TRUE, ignore.stderr = TRUE) == 0) {
    options(editor = "nano")
  } else if (system("which vim > /dev/null 2>&1", ignore.stdout = TRUE, ignore.stderr = TRUE) == 0) {
    options(editor = "vim")
  } else {
    tryCatch({ options(editor = NULL) }, error = function(e) {})
  }
}, error = function(e) {
  tryCatch({ options(editor = NULL) }, error = function(e2) {})
})
# Create PDF visualizations for clustering results
#
# Generates comprehensive PDF report with all visualizations

.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(utils)  # Explicitly load utils for read.csv, write.csv, etc.
  library(stats)  # Explicitly load stats for dist, prcomp, etc.
  library(optparse)
  library(dplyr)
  library(ggplot2)
  library(Rtsne)
  library(factoextra)
  library(pheatmap)
  library(gridExtra)
  library(grid)
  library(dendextend)
})

# Create consistent color mapping function
# This ensures the same cluster ID always gets the same color across all visualizations
create_cluster_color_map <- function(cluster_ids) {
  unique_clusters <- sort(unique(cluster_ids))
  n_clusters <- length(unique_clusters)
  
  # Use a colorblind-friendly palette with enough distinct colors
  if (n_clusters <= 8) {
    # Use RColorBrewer Set2 or similar for small number of clusters
    base_colors <- c("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", 
                     "#FFD92F", "#E5C494", "#B3B3B3")
    colors <- base_colors[1:n_clusters]
  } else {
    # Use rainbow for larger numbers, but ensure consistent ordering
    colors <- rainbow(n_clusters)
  }
  
  # Create named vector mapping cluster ID to color
  color_map <- setNames(colors, as.character(unique_clusters))
  return(color_map)
}

option_list <- list(
  make_option(c("-f", "--features"), type="character", default=NULL,
              help="Input CSV file with bout features"),
  make_option(c("-c", "--clusters"), type="character", default=NULL,
              help="Input CSV file with cluster assignments (required)"),
  make_option(c("-o", "--output-dir"), type="character", default=".",
              help="Output directory"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Get method from environment variable (passed from Python due to getopt limitation)
method_value <- Sys.getenv("CLUSTER_METHOD", unset = "")
# Fallback: try to infer from cluster file name
if (method_value == "" || is.null(method_value) || nchar(method_value) == 0) {
  if (!is.null(opt$clusters)) {
    cluster_basename <- basename(opt$clusters)
    if (grepl("_hierarchical", cluster_basename, ignore.case=TRUE)) {
      method_value <- "hierarchical"
    } else if (grepl("_kmeans", cluster_basename, ignore.case=TRUE)) {
      method_value <- "kmeans"
    } else if (grepl("_dbscan", cluster_basename, ignore.case=TRUE)) {
      method_value <- "dbscan"
    } else if (grepl("_bsoid", cluster_basename, ignore.case=TRUE)) {
      method_value <- "bsoid"
    }
  }
}
opt$method <- if (method_value != "" && !is.null(method_value) && nchar(method_value) > 0) method_value else NULL

# Check required arguments
if (is.null(opt$features)) {
  stop("Error: --features is required. Use --help for usage information.")
}
if (is.null(opt$clusters)) {
  stop("Error: --clusters is required. Use --help for usage information.")
}
# Method is optional - can be inferred from filename if not provided

# Load and prepare data (reuse functions from visualize_clusters.R)
# Get script directory
cmd_args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args_full, value = TRUE)
if (length(file_arg) > 0) {
  script_path_full <- sub("^--file=", "", file_arg)
  script_dir <- dirname(normalizePath(script_path_full))
} else {
  script_dir <- getwd()
}
script_path <- file.path(script_dir, "visualize_clusters.R")
if (!file.exists(script_path)) {
  # Try relative to current working directory
  script_path <- file.path("BoutAnalysisScripts", "scripts", "visualization", "visualize_clusters.R")
  if (!file.exists(script_path)) {
    script_path <- "visualize_clusters.R"  # Last fallback
  }
}
source(script_path, local=TRUE)

main <- function() {
  cat(sprintf("Loading features: %s\n", opt$features))
  features_df <- read.csv(opt$features, stringsAsFactors=FALSE)
  
  cat(sprintf("Loading clusters: %s\n", opt$clusters))
  clusters_df <- read.csv(opt$clusters, stringsAsFactors=FALSE)
  
  df <- merge(features_df, clusters_df[, c("bout_id", "cluster_id")], 
             by="bout_id", all.x=FALSE)
  
  prepared <- prepare_features(df)
  metadata_df <- prepared$metadata
  X_scaled <- prepared$features
  feature_names <- prepared$feature_names
  
  labels <- clusters_df$cluster_id[match(df$bout_id, clusters_df$bout_id)]
  
  # Get method name (from environment or inferred)
  # Check if opt$method was set correctly
  if (is.null(opt$method) || opt$method == "" || is.na(opt$method)) {
    # Try to infer from filename again
    cluster_basename <- basename(opt$clusters)
    if (grepl("_hierarchical", cluster_basename, ignore.case=TRUE)) {
      method_name <- "hierarchical"
    } else if (grepl("_kmeans", cluster_basename, ignore.case=TRUE)) {
      method_name <- "kmeans"
    } else if (grepl("_dbscan", cluster_basename, ignore.case=TRUE)) {
      method_name <- "dbscan"
    } else if (grepl("_bsoid", cluster_basename, ignore.case=TRUE)) {
      method_name <- "bsoid"
    } else {
      method_name <- "unknown"
    }
  } else {
    method_name <- opt$method
  }
  
  # Create consistent color mapping for all visualizations
  cluster_color_map <- create_cluster_color_map(labels)
  
  # Store method_name and color_map for use in other sections
  assign("method_name", method_name, envir=environment())
  assign("cluster_color_map", cluster_color_map, envir=environment())
  
  # Create PDF
  pdf_file <- file.path(opt$`output-dir`, sprintf("clustering_%s_report.pdf", method_name))
  pdf(pdf_file, width=12, height=10)
  
  cat("Creating visualizations...\n")
  
  # Page 0: Dendrogram (for hierarchical clustering only)
  if (method_name == "hierarchical") {
    tryCatch({
      # Reconstruct hierarchical clustering to get dendrogram
      dist_matrix <- dist(X_scaled)
      hc <- hclust(dist_matrix, method="ward.D2")
      dend <- as.dendrogram(hc)
      
      # Color branches by cluster - use consistent color mapping
      n_clusters <- length(unique(labels))
      
      # Get cluster assignments from cutree to match dendrogram structure
      labels_from_dend <- cutree(hc, k=n_clusters)
      
      # Map colors based on actual cluster assignments, ensuring consistency
      # Create a vector of colors for each sample based on its cluster assignment
      sample_colors <- cluster_color_map[as.character(labels_from_dend)]
      
      # Color the dendrogram using dendextend if available
      if (requireNamespace("dendextend", quietly=TRUE)) {
        # Use the color map values in order of sorted cluster IDs
        dend_colored <- dendextend::color_branches(dend, k=n_clusters, 
                                                   col=as.vector(cluster_color_map))
        
        # Plot dendrogram
        par(mar=c(8, 4, 4, 2) + 0.1)
        plot(dend_colored, 
             main=sprintf("Hierarchical Clustering Dendrogram - %d Clusters", n_clusters),
             xlab="Bout Index", 
             ylab="Height",
             leaflab="none",  # Don't show all leaf labels (too many)
             sub="")
        
        # Add legend with consistent colors
        unique_clusters <- sort(unique(labels))
        legend_colors <- cluster_color_map[as.character(unique_clusters)]
        legend("topright", 
               legend=paste("Cluster", unique_clusters),
               fill=as.vector(legend_colors),
               cex=0.8,
               title="Clusters")
        
        # Add explanation text at bottom
        mtext("Interpretation: This dendrogram shows the hierarchical clustering tree structure. Each branch represents a group of similar bouts. The height axis indicates the distance at which clusters merge - lower heights indicate more similar bouts. Branches are colored by final cluster assignment. Clusters that merge at lower heights are more similar to each other.", 
              side=1, line=5, cex=0.7, adj=0, padj=1, outer=FALSE)
      } else {
        # Fallback: use factoextra
        p_dend <- factoextra::fviz_dend(hc, k=n_clusters, 
                                        cex=0.3, 
                                        k_colors=as.vector(cluster_color_map),
                                        color_labels_by_k=TRUE,
                                        show_labels=FALSE,
                                        main=sprintf("Hierarchical Clustering Dendrogram - %d Clusters", n_clusters))
        print(p_dend)
      }
    }, error = function(e) {
      cat(sprintf("Warning: Could not create dendrogram: %s\n", e$message))
      # Create a simple text plot as fallback
      plot.new()
      text(0.5, 0.5, sprintf("Dendrogram visualization not available\nError: %s", e$message), cex=1.0)
    })
  }
  
  # Page 1: PCA
  pca_result <- prcomp(X_scaled, scale.=FALSE)
  pca_df <- data.frame(
    PC1 = pca_result$x[, 1],
    PC2 = pca_result$x[, 2],
    cluster = as.factor(labels)
  )
  p1 <- ggplot(pca_df, aes(x=PC1, y=PC2, color=cluster)) +
    geom_point(alpha=0.6, size=2) +
    scale_color_manual(values=cluster_color_map, name="Cluster") +
    labs(title=sprintf("PCA Visualization - %s", method_name),
         x=sprintf("PC1 (%.1f%% variance)", summary(pca_result)$importance[2,1]*100),
         y=sprintf("PC2 (%.1f%% variance)", summary(pca_result)$importance[2,2]*100),
         caption="Interpretation: Principal Component Analysis (PCA) reduces high-dimensional feature space to 2D. Each point represents a bout, colored by cluster. PC1 and PC2 are linear combinations of original features that capture the most variance. Points close together have similar feature patterns. Clusters should form distinct groups if clustering is meaningful.") +
    theme_minimal() +
    theme(plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
  print(p1)
  
  # Page 2: t-SNE
  if (nrow(X_scaled) <= 1000) {
    tsne_result <- Rtsne(X_scaled, dims=2, perplexity=min(30, max(5, floor((nrow(X_scaled)-1)/3))), verbose=FALSE)
    tsne_df <- data.frame(
      tSNE1 = tsne_result$Y[, 1],
      tSNE2 = tsne_result$Y[, 2],
      cluster = as.factor(labels)
    )
    p2 <- ggplot(tsne_df, aes(x=tSNE1, y=tSNE2, color=cluster)) +
      geom_point(alpha=0.6, size=2) +
      scale_color_manual(values=cluster_color_map, name="Cluster") +
      labs(title=sprintf("t-SNE Visualization - %s", method_name),
           caption="Interpretation: t-SNE (t-distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique that preserves local neighborhood structure. Points close in the original high-dimensional space remain close in 2D. Clusters should appear as distinct groups. Unlike PCA, t-SNE distances are not directly interpretable, but it often reveals cluster structure better than PCA.") +
      theme_minimal() +
      theme(plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
    print(p2)
  }
  
  # Page 3: Cluster sizes
  cluster_counts <- table(labels)
  cluster_df <- data.frame(
    cluster = names(cluster_counts),
    count = as.numeric(cluster_counts)
  )
  # Ensure cluster_df is ordered and has consistent colors
  cluster_df$cluster <- factor(cluster_df$cluster, levels=sort(unique(labels)))
  cluster_fill_colors <- cluster_color_map[as.character(cluster_df$cluster)]
  
  p3 <- ggplot(cluster_df, aes(x=cluster, y=count, fill=cluster)) +
    geom_bar(stat="identity") +
    scale_fill_manual(values=as.vector(cluster_fill_colors), guide="none") +
    labs(title="Cluster Sizes", 
         x="Cluster ID", 
         y="Number of Bouts",
         caption="Interpretation: This bar chart shows the number of bouts assigned to each cluster. Balanced clusters (similar sizes) are generally preferred, but natural behavior patterns may result in uneven distributions. Very small clusters may represent rare behaviors, while large clusters may indicate common behavior patterns.") +
    theme_minimal() +
    theme(legend.position="none",
          plot.caption = element_text(hjust=0, size=8, margin=margin(t=10)))
  print(p3)
  
  # Page 4: Feature distributions (top features)
  if (length(feature_names) > 0) {
    top_features <- head(feature_names, min(12, length(feature_names)))
    feature_plots <- list()
    for (i in seq_along(top_features)) {
      feat <- top_features[i]
      if (feat %in% colnames(df)) {
        # Ensure cluster_id is factor with consistent ordering and colors
        df$cluster_id_factor <- factor(df$cluster_id, levels=sort(unique(labels)))
        cluster_fill_colors_feat <- cluster_color_map[as.character(levels(df$cluster_id_factor))]
        
        p <- ggplot(df, aes_string(x="cluster_id_factor", y=feat, fill="cluster_id_factor")) +
          geom_boxplot(alpha=0.7) +
          scale_fill_manual(values=as.vector(cluster_fill_colors_feat), guide="none") +
          labs(title=feat, x="Cluster", y="Value",
               caption="Interpretation: Boxplots show the distribution of feature values across clusters. The box shows the interquartile range (IQR), the line is the median, and whiskers extend to 1.5×IQR. Differences in median values and distributions between clusters indicate which features distinguish clusters. Overlapping boxes suggest similar feature values across clusters.") +
          theme_minimal() +
          theme(legend.position="none", 
                axis.text.x=element_text(angle=45, hjust=1),
                plot.caption = element_text(hjust=0, size=7, margin=margin(t=8)))
        feature_plots[[length(feature_plots)+1]] <- p
      }
    }
    if (length(feature_plots) > 0) {
      n_cols <- 3
      n_rows <- ceiling(length(feature_plots) / n_cols)
      do.call(grid.arrange, c(feature_plots, ncol=n_cols))
    }
  }
  
  # Page 5: Cluster heatmap
  unique_labels <- sort(unique(labels))
  # Filter out noise clusters (0 or negative) for heatmap if DBSCAN
  if (method_name == "dbscan" || method_name == "bsoid") {
    valid_clusters <- unique_labels[unique_labels > 0]
    if (length(valid_clusters) == 0) {
      valid_clusters <- unique_labels  # Fallback to all clusters
    }
  } else {
    valid_clusters <- unique_labels
  }
  
  cluster_means <- matrix(0, nrow=length(valid_clusters), ncol=ncol(X_scaled))
  rownames(cluster_means) <- paste0("Cluster ", valid_clusters)
  
  for (i in seq_along(valid_clusters)) {
    cluster_id <- valid_clusters[i]
    cluster_mask <- labels == cluster_id
    if (sum(cluster_mask) > 0) {
      cluster_means[i, ] <- colMeans(X_scaled[cluster_mask, , drop=FALSE], na.rm=TRUE)
    }
  }
  
  # Use top features for heatmap (only if we have valid clusters)
  if (nrow(cluster_means) > 0 && ncol(cluster_means) > 0) {
    n_features_heatmap <- min(50, ncol(X_scaled))
    top_var_features <- order(apply(X_scaled, 2, var, na.rm=TRUE), decreasing=TRUE)[1:n_features_heatmap]
    
    method_name <- if (!is.null(opt$method) && opt$method != "") opt$method else "unknown"
    heatmap_title <- sprintf("Cluster Feature Heatmap - %s", method_name)
    
    # Prepare heatmap data
    heatmap_data <- cluster_means[, top_var_features, drop=FALSE]
    
    # Only cluster columns if we have more than 1 feature, and only if we have more than 1 cluster
    cluster_cols <- ncol(heatmap_data) > 1 && nrow(heatmap_data) > 1
    
      # Use tryCatch to handle edge cases
      tryCatch({
        pheatmap(t(heatmap_data),
                main=heatmap_title,
                cluster_rows=FALSE, cluster_cols=cluster_cols,
                show_colnames=FALSE, fontsize=8)
      }, error = function(e) {
        # Fallback: simple heatmap without clustering
        pheatmap(t(heatmap_data),
                main=heatmap_title,
                cluster_rows=FALSE, cluster_cols=FALSE,
                show_colnames=FALSE, fontsize=8)
      })
      
      # Add explanation text on a new plot page after heatmap
      plot.new()
      text(0.5, 0.5, "Interpretation: This heatmap shows the mean feature values for each cluster (rows) across the top variable features (columns). Colors represent standardized values: red = high, blue = low, white = average. Clusters with similar color patterns share similar feature profiles. Features are ordered by hierarchical clustering to group similar features together.", 
           cex=0.8, adj=c(0.5, 0.5))
  } else {
    plot.new()
    text(0.5, 0.5, "No valid clusters for heatmap visualization", cex=1.2)
  }
  
  # Summary statistics
  cluster_counts <- table(labels)
  method_name <- if (!is.null(opt$method)) opt$method else "unknown"
  summary_text <- sprintf(
    "Clustering Method: %s\n\nTotal Bouts: %d\nNumber of Clusters: %d\n\nCluster Sizes:\n%s",
    method_name,
    nrow(df),
    length(unique_labels),
    paste(sprintf("  Cluster %s: %d bouts", names(cluster_counts), cluster_counts), collapse="\n")
  )
  
  grid.text(summary_text, x=0.5, y=0.5, gp=gpar(fontsize=12))
  
  dev.off()
  
  cat(sprintf("✓ PDF saved: %s\n", pdf_file))
  
  # Cleanup: Remove Rplots.pdf if it was accidentally created
  if (file.exists("Rplots.pdf")) {
    file.remove("Rplots.pdf")
    cat("Note: Removed Rplots.pdf from working directory\n")
  }
}

if (!interactive()) {
  main()
}

