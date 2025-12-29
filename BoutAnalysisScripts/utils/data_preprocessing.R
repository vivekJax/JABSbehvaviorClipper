# Data Preprocessing Utilities
# Functions for cleaning and preparing data for clustering and outlier detection

#' Handle missing values in a data frame
#'
#' @param df Data frame with potential missing values
#' @param strategy Strategy for handling missing values: "mean" (imputation), "drop" (remove rows/columns)
#' @return Data frame with missing values handled
handle_missing_values <- function(df, strategy = "mean") {
  if (strategy == "mean") {
    # Mean imputation for numeric columns
    numeric_cols <- sapply(df, is.numeric)
    for (col in names(df)[numeric_cols]) {
      if (any(is.na(df[[col]]))) {
        mean_val <- mean(df[[col]], na.rm = TRUE)
        if (!is.na(mean_val)) {
          df[[col]][is.na(df[[col]])] <- mean_val
        } else {
          # If all values are NA, set to 0
          df[[col]][is.na(df[[col]])] <- 0
        }
      }
    }
  } else if (strategy == "drop") {
    # Remove rows with any missing values
    df <- df[complete.cases(df), ]
  }
  
  return(df)
}

#' Remove constant/zero-variance features
#'
#' @param df Data frame
#' @param numeric_only If TRUE, only check numeric columns
#' @return Data frame with constant features removed
remove_constant_features <- function(df, numeric_only = TRUE) {
  if (numeric_only) {
    numeric_cols <- sapply(df, is.numeric)
    cols_to_check <- names(df)[numeric_cols]
  } else {
    cols_to_check <- names(df)
  }
  
  constant_cols <- c()
  for (col in cols_to_check) {
    if (is.numeric(df[[col]])) {
      # Check for zero variance
      if (var(df[[col]], na.rm = TRUE) == 0 || 
          length(unique(df[[col]][!is.na(df[[col]])])) <= 1) {
        constant_cols <- c(constant_cols, col)
      }
    } else {
      # For non-numeric, check if all values are the same
      if (length(unique(df[[col]][!is.na(df[[col]])])) <= 1) {
        constant_cols <- c(constant_cols, col)
      }
    }
  }
  
  if (length(constant_cols) > 0) {
    df <- df[, !names(df) %in% constant_cols, drop = FALSE]
  }
  
  return(df)
}

#' Standardize features (z-score normalization)
#'
#' @param df Data frame with numeric features
#' @param exclude_cols Column names to exclude from standardization (e.g., metadata columns)
#' @return List with:
#'   - scaled_df: Standardized data frame
#'   - means: Vector of means used for centering
#'   - sds: Vector of standard deviations used for scaling
standardize_features <- function(df, exclude_cols = c("bout_id", "video_name", "animal_id", 
                                                      "start_frame", "end_frame", "behavior")) {
  # Identify numeric columns to scale
  numeric_cols <- sapply(df, is.numeric)
  cols_to_scale <- names(df)[numeric_cols & !names(df) %in% exclude_cols]
  
  if (length(cols_to_scale) == 0) {
    return(list(scaled_df = df, means = numeric(0), sds = numeric(0)))
  }
  
  # Calculate means and standard deviations
  means <- sapply(df[cols_to_scale], mean, na.rm = TRUE)
  sds <- sapply(df[cols_to_scale], sd, na.rm = TRUE)
  
  # Handle zero standard deviations (constant features)
  sds[sds == 0 | is.na(sds)] <- 1
  
  # Create scaled data frame
  scaled_df <- df
  for (col in cols_to_scale) {
    scaled_df[[col]] <- (df[[col]] - means[col]) / sds[col]
  }
  
  return(list(scaled_df = scaled_df, means = means, sds = sds))
}

#' Handle infinite and NaN values
#'
#' @param df Data frame
#' @return Data frame with infinite and NaN values replaced
handle_non_finite <- function(df) {
  numeric_cols <- sapply(df, is.numeric)
  
  for (col in names(df)[numeric_cols]) {
    # Replace Inf and -Inf with NA, then with 0
    df[[col]][is.infinite(df[[col]])] <- NA
    df[[col]][is.nan(df[[col]])] <- NA
    
    # Replace remaining NA with 0
    df[[col]][is.na(df[[col]])] <- 0
  }
  
  return(df)
}

#' Prepare features for clustering/analysis
#'
#' Complete preprocessing pipeline:
#' 1. Handle missing values
#' 2. Remove constant features
#' 3. Handle non-finite values
#' 4. Standardize features
#'
#' @param df Data frame with features
#' @param exclude_cols Columns to exclude from preprocessing
#' @param missing_strategy Strategy for missing values: "mean" or "drop"
#' @return List with:
#'   - processed_df: Preprocessed data frame
#'   - feature_cols: Names of feature columns (excluding metadata)
#'   - scaling_info: List with means and sds for scaling
prepare_features <- function(df, exclude_cols = c("bout_id", "video_name", "animal_id", 
                                                   "start_frame", "end_frame", "behavior"),
                             missing_strategy = "mean") {
  # Step 1: Handle missing values
  df <- handle_missing_values(df, strategy = missing_strategy)
  
  # Step 2: Remove constant features
  df <- remove_constant_features(df, numeric_only = TRUE)
  
  # Step 3: Handle non-finite values
  df <- handle_non_finite(df)
  
  # Step 4: Standardize features
  scaling_result <- standardize_features(df, exclude_cols = exclude_cols)
  scaled_df <- scaling_result$scaled_df
  
  # Identify feature columns (numeric columns excluding metadata)
  numeric_cols <- sapply(scaled_df, is.numeric)
  feature_cols <- names(scaled_df)[numeric_cols & !names(scaled_df) %in% exclude_cols]
  
  return(list(
    processed_df = scaled_df,
    feature_cols = feature_cols,
    scaling_info = list(means = scaling_result$means, sds = scaling_result$sds)
  ))
}

