#!/usr/bin/env python3
"""
Complete Analysis Pipeline

Order of operations:
1. Make behavior video of all bouts
2. Detect outliers using multiple methods + consensus
3. Make video of outliers that are removed
4. Use PCA for dimensionality reduction (95% variance)
5. Cluster with PCA features using multiple methods
6. For each method: PDF visualizations + cluster videos
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(cmd, description, check=True, env=None):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print()
    
    result = subprocess.run(cmd, shell=isinstance(cmd, str), check=check, env=env)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        if check:
            sys.exit(result.returncode)
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete analysis pipeline with specified order of operations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Order of operations:
  1. Extract features and make video of all bouts
  2. Detect outliers (multiple methods + consensus)
  3. Make video of outliers
  4. Apply PCA (95% variance) and remove outliers
  5. Cluster with multiple methods
  6. Generate PDF visualizations and cluster videos
        """
    )
    
    parser.add_argument('-b', '--behavior', default='turn_left',
                       help='Behavior name (default: turn_left)')
    parser.add_argument('-a', '--annotations-dir', default='../jabs/annotations',
                       help='Directory containing annotation JSON files (default: ../jabs/annotations)')
    parser.add_argument('-f', '--features-dir', default='../jabs/features',
                       help='Base directory for feature HDF5 files (default: ../jabs/features)')
    parser.add_argument('-v', '--video-dir', default='..',
                       help='Directory containing video files (default: parent directory)')
    parser.add_argument('-o', '--output-dir', default='BoutResults',
                       help='Output directory for all results (default: BoutResults)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU cores - 1)')
    parser.add_argument('--pca-variance', type=float, default=0.95,
                       help='Proportion of variance to retain in PCA (default: 0.95)')
    parser.add_argument('--outlier-threshold', default='auto',
                       help="Outlier threshold: 'auto' (top 5%%), 'topN', or percentile")
    parser.add_argument('--outlier-top-n', type=int, default=None,
                       help='Number of top outliers (used with threshold=topN)')
    parser.add_argument('--consensus-min', type=int, default=2,
                       help='Minimum methods that must agree for consensus (default: 2)')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature extraction (use existing CSV)')
    parser.add_argument('--skip-all-bouts-video', action='store_true',
                       help='Skip all bouts video generation')
    parser.add_argument('--skip-outliers', action='store_true',
                       help='Skip outlier detection')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='Skip clustering')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation even if cache exists')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Determine workers
    if args.workers is None:
        import multiprocessing
        args.workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'outliers'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'clustering'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'cache'), exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Complete Analysis Pipeline")
    print(f"{'='*60}")
    print(f"Behavior: {args.behavior}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    features_file = os.path.join(args.output_dir, 'bout_features.csv')
    
    # Step 1: Extract features
    if not args.skip_features:
        cmd = [
            'python3', 'scripts/extract_bout_features.py',
            '--behavior', args.behavior,
            '--annotations-dir', args.annotations_dir,
            '--features-dir', args.features_dir,
            '--output', features_file,
            '--cache-dir', os.path.join(args.output_dir, 'cache'),
            '--workers', str(args.workers)
        ]
        if args.force_recompute:
            cmd.append('--force-recompute')
        if args.verbose:
            cmd.append('--verbose')
        
        run_command(cmd, "Step 1: Extracting bout features", check=True)
    else:
        if not os.path.exists(features_file):
            print(f"Error: Features file not found: {features_file}")
            sys.exit(1)
        print(f"Skipping feature extraction. Using: {features_file}")
    
    # Step 2: Make video of all bouts
    if not args.skip_all_bouts_video:
        all_bouts_video = os.path.join(args.output_dir, 'videos', 'all_bouts.mp4')
        cmd = [
            'python3', 'scripts/generate_bouts_video.py',
            '--behavior', args.behavior,
            '--annotations-dir', args.annotations_dir,
            '--video-dir', args.video_dir,
            '--output', all_bouts_video,
            '--workers', str(args.workers)
        ]
        if args.verbose:
            cmd.append('--verbose')
        
        run_command(cmd, "Step 2: Creating video of all bouts", check=False)
    
    # Step 3: Detect outliers (multiple methods + consensus)
    if not args.skip_outliers:
        outlier_dir = os.path.join(args.output_dir, 'outliers')
        cmd = [
            'Rscript', 'BoutAnalysisScripts/scripts/core/detect_outliers_consensus.R',
            '--features', features_file,
            '--output-dir', outlier_dir,
            '--pca-variance', str(args.pca_variance),
            '--threshold', args.outlier_threshold
            # Note: consensus-min is hardcoded to 2 in the R script (getopt limitation)
        ]
        if args.outlier_top_n:
            cmd.extend(['--top-n', str(args.outlier_top_n)])
        if args.verbose:
            cmd.append('--verbose')
        
        run_command(cmd, "Step 3: Detecting outliers (multiple methods + consensus)", check=True)
        
        # Step 4: Make video of outliers
        outlier_csv = os.path.join(outlier_dir, 'consensus_outliers.csv')
        if os.path.exists(outlier_csv):
            outlier_video = os.path.join(args.output_dir, 'videos', 'outliers.mp4')
            
            # Create annotation files for outliers
            import pandas as pd
            outliers_df = pd.read_csv(outlier_csv)
            
            temp_ann_dir = os.path.join(outlier_dir, 'temp_annotations')
            os.makedirs(temp_ann_dir, exist_ok=True)
            
            # Group by video
            for video_name in outliers_df['video_name'].unique():
                video_bouts = outliers_df[outliers_df['video_name'] == video_name]
                animals = video_bouts['animal_id'].unique()
                labels = {}
                
                for animal_id in animals:
                    animal_bouts = video_bouts[video_bouts['animal_id'] == animal_id]
                    behavior_bouts = []
                    
                    for _, bout in animal_bouts.iterrows():
                        behavior_bouts.append({
                            'start': int(bout['start_frame']),
                            'end': int(bout['end_frame']),
                            'present': True
                        })
                    
                    labels[str(animal_id)] = {args.behavior: behavior_bouts}
                
                annotation_data = {
                    'file': video_name,
                    'labels': labels
                }
                
                video_basename = os.path.splitext(video_name)[0]
                json_file = os.path.join(temp_ann_dir, f"{video_basename}.json")
                with open(json_file, 'w') as f:
                    json.dump(annotation_data, f, indent=2)
            
            # Generate outlier video
            cmd = [
                'python3', 'scripts/generate_bouts_video.py',
                '--behavior', args.behavior,
                '--annotations-dir', temp_ann_dir,
                '--video-dir', args.video_dir,
                '--output', outlier_video,
                '--workers', str(args.workers)
            ]
            if args.verbose:
                cmd.append('--verbose')
            
            run_command(cmd, "Step 4: Creating video of outliers", check=False)
    
    # Step 5: Remove outliers and apply PCA
    if not args.skip_clustering:
        outlier_csv = os.path.join(args.output_dir, 'outliers', 'consensus_outliers.csv')
        
        if os.path.exists(outlier_csv):
            import pandas as pd
            df = pd.read_csv(features_file)
            outliers_df = pd.read_csv(outlier_csv)
            
            # Remove outliers
            df_clean = df[~df['bout_id'].isin(outliers_df['bout_id'])]
            print(f"\nRemoved {len(outliers_df)} outliers, {len(df_clean)} bouts remaining")
            
            # Save clean features
            features_clean_file = os.path.join(args.output_dir, 'clustering', 'bout_features_clean.csv')
            df_clean.to_csv(features_clean_file, index=False)
            
            # Apply PCA using R
            pca_features_file = os.path.join(args.output_dir, 'clustering', 'bout_features_pca.csv')
            cmd = [
                'Rscript', '-e',
                f'''
                library(dplyr)
                df <- read.csv("{features_clean_file}", stringsAsFactors=FALSE)
                metadata_cols <- c("bout_id", "video_name", "animal_id", "start_frame", 
                                 "end_frame", "behavior", "duration_frames")
                feature_cols <- setdiff(colnames(df), metadata_cols)
                X <- as.matrix(df[, feature_cols])
                X[!is.finite(X)] <- 0
                # Remove constant/zero variance columns
                col_vars <- apply(X, 2, var, na.rm=TRUE)
                non_zero_var_cols <- col_vars > 1e-10
                X <- X[, non_zero_var_cols, drop=FALSE]
                cat(sprintf("Removed %d constant columns, %d features remaining\\n",
                           sum(!non_zero_var_cols), sum(non_zero_var_cols)))
                X_scaled <- scale(X)
                X_scaled[!is.finite(X_scaled)] <- 0
                pca_result <- prcomp(X_scaled, scale.=FALSE)
                variance_explained <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
                n_components <- which(variance_explained >= {args.pca_variance})[1]
                if (is.na(n_components)) n_components <- length(variance_explained)
                X_pca <- pca_result$x[, 1:n_components, drop=FALSE]
                cat(sprintf("PCA: %d features -> %d components (%.1f%% variance)\\n",
                           ncol(X_scaled), n_components, variance_explained[n_components] * 100))
                pca_df <- cbind(df[, metadata_cols], as.data.frame(X_pca))
                write.csv(pca_df, "{pca_features_file}", row.names=FALSE)
                '''
            ]
            run_command(cmd, "Step 5: Applying PCA (95% variance)", check=True)
        else:
            print("Warning: No outliers file found, using all features")
            pca_features_file = features_file
        
        # Step 6: Cluster with multiple methods
        clustering_methods = ['kmeans', 'hierarchical', 'dbscan']
        
        for method in clustering_methods:
            print(f"\n{'='*60}")
            print(f"Step 6: Clustering with {method}")
            print(f"{'='*60}")
            
            cluster_dir = os.path.join(args.output_dir, 'clustering', method)
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Run clustering
            cluster_file = os.path.join(cluster_dir, f'cluster_assignments_{method}.csv')
            cmd = [
                'Rscript', 'BoutAnalysisScripts/scripts/core/cluster_bouts.R',
                '--input', pca_features_file,
                '--method', method,
                '--output-dir', cluster_dir,
                '--ncores', str(args.workers)
            ]
            if args.verbose:
                cmd.append('--verbose')
            
            if run_command(cmd, f"Clustering with {method}", check=False):
                if os.path.exists(cluster_file):
                    # Generate PDF visualizations
                    # Note: --method is passed via environment variable due to getopt limitation
                    env = os.environ.copy()
                    env['CLUSTER_METHOD'] = method
                    pdf_cmd = [
                        'Rscript', 'BoutAnalysisScripts/scripts/visualization/visualize_clusters_pdf.R',
                        '--features', pca_features_file,
                        '--clusters', cluster_file,
                        '--output-dir', cluster_dir
                    ]
                    if args.verbose:
                        pdf_cmd.append('--verbose')
                    
                    # Run with environment variable for method
                    run_command(pdf_cmd, f"Generating PDF visualizations for {method}", check=False, env=env)
                    
                    # Generate cluster videos
                    video_cmd = [
                        'Rscript', 'BoutAnalysisScripts/scripts/video/generate_cluster_videos.R',
                        '--clusters', cluster_file,
                        '--method', method,
                        '--annotations-dir', args.annotations_dir,
                        '--video-dir', args.video_dir,
                        '--output-dir', os.path.join(args.output_dir, 'videos', method),
                        '--behavior', args.behavior,
                        '--workers', str(args.workers)
                    ]
                    if args.verbose:
                        video_cmd.append('--verbose')
                    
                    run_command(video_cmd, f"Generating cluster videos for {method}", check=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("Analysis Pipeline Complete!")
    print(f"{'='*60}")
    print(f"\nAll results organized in: {args.output_dir}/")
    print("\nGenerated files:")
    print(f"  - Features: {features_file}")
    if not args.skip_all_bouts_video:
        print(f"  - All bouts video: {args.output_dir}/videos/all_bouts.mp4")
    if not args.skip_outliers:
        print(f"  - Outlier video: {args.output_dir}/videos/outliers.mp4")
        print(f"  - Consensus outliers: {args.output_dir}/outliers/consensus_outliers.csv")
    if not args.skip_clustering:
        print(f"  - PCA features: {args.output_dir}/clustering/bout_features_pca.csv")
        print(f"  - Cluster assignments: {args.output_dir}/clustering/*/cluster_assignments_*.csv")
        print(f"  - PDF reports: {args.output_dir}/clustering/*/clustering_*_report.pdf")
        print(f"  - Cluster videos: {args.output_dir}/videos/*/")
    print(f"\nAll output is organized in '{args.output_dir}/' directory for cleanliness.")
    print()


if __name__ == '__main__':
    main()

