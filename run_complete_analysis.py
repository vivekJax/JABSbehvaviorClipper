#!/usr/bin/env python3
# Note: If Anaconda Python has segfault issues, use system Python:
# /usr/bin/python3 run_complete_analysis.py ...
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
import time
from pathlib import Path
import shutil
from datetime import datetime


def get_python_cmd():
    """Get working Python command, preferring system Python if Anaconda has issues."""
    # Try system Python first (usually more stable)
    system_python = '/usr/bin/python3'
    # Test if it works
    try:
        result = subprocess.run([system_python, '-c', 'import numpy, h5py, pandas'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            return system_python
    except:
        pass
    # Fallback to default python3
    return 'python3'


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_progress_header(step_num, total_steps, description, elapsed_time=None):
    """Print a formatted progress header."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    progress_pct = int((step_num / total_steps) * 100) if total_steps > 0 else 0
    progress_bar = "█" * (progress_pct // 2) + "░" * (50 - progress_pct // 2)
    
    print(f"\n{'='*70}")
    print(f"[{timestamp}] Step {step_num}/{total_steps} ({progress_pct}%) - {description}")
    if elapsed_time:
        print(f"  Elapsed: {format_time(elapsed_time)}")
    print(f"  [{progress_bar}] {progress_pct}%")
    print(f"{'='*70}")


def run_command(cmd, description, check=True, env=None, step_num=None, total_steps=None, elapsed_time=None, verbose=False):
    """Run a shell command and handle errors with progress indication."""
    start_time = time.time()
    
    if step_num and total_steps:
        print_progress_header(step_num, total_steps, description, elapsed_time)
    else:
        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
        print(f"{'='*70}")
    
    # Show command (truncated if long)
    cmd_str = ' '.join(cmd) if isinstance(cmd, list) else str(cmd)
    if len(cmd_str) > 100:
        cmd_str = cmd_str[:97] + "..."
    print(f"  Command: {cmd_str}")
    print()
    
    # Run command
    result = subprocess.run(
        cmd, 
        shell=isinstance(cmd, str), 
        check=False, 
        env=env,
        stdout=subprocess.PIPE if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
        text=True
    )
    
    duration = time.time() - start_time
    
    # Show last few lines of output if not verbose
    if not verbose and result.stdout:
        output_lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
        if output_lines:
            print("  Recent output:")
            for line in output_lines[-5:]:  # Last 5 lines
                print(f"    {line}")
    
    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {description}")
        print(f"     Exit code: {result.returncode}, Duration: {format_time(duration)}")
        if check:
            print(f"\n  Pipeline stopped due to error.")
            sys.exit(result.returncode)
        return False
    
    print(f"\n  ✓ COMPLETED: {description} ({format_time(duration)})")
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
    
    # Count total steps for progress tracking
    total_steps = 0
    if not args.skip_features:
        total_steps += 1
    if not args.skip_all_bouts_video:
        total_steps += 1
    if not args.skip_outliers:
        total_steps += 2  # Detection + video
    if not args.skip_clustering:
        total_steps += 1  # PCA
        total_steps += len(['hierarchical', 'bsoid']) * 3  # Clustering + PDF + videos per method
        total_steps += 1  # Report generation
    
    current_step = 0
    pipeline_start_time = time.time()
    
    print(f"\n{'='*70}")
    print("COMPLETE ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print(f"Behavior: {args.behavior}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output_dir}")
    print(f"Total Steps: {total_steps}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Get working Python command
    python_cmd = get_python_cmd()
    print(f"Using Python: {python_cmd}\n")
    
    features_file = os.path.join(args.output_dir, 'bout_features.csv')
    
    # Step 1: Extract features
    if not args.skip_features:
        current_step += 1
        elapsed = time.time() - pipeline_start_time
        cmd = [
            python_cmd, 'scripts/extract_bout_features.py',
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
        
        run_command(cmd, "Extracting Bout Features", check=True, 
                   step_num=current_step, total_steps=total_steps, elapsed_time=elapsed, verbose=args.verbose)
    else:
        if not os.path.exists(features_file):
            print(f"Error: Features file not found: {features_file}")
            sys.exit(1)
        print(f"⏭  Skipping feature extraction. Using: {features_file}")
    
    # Step 2: Make video of all bouts
    if not args.skip_all_bouts_video:
        current_step += 1
        elapsed = time.time() - pipeline_start_time
        all_bouts_video = os.path.join(args.output_dir, 'videos', 'all_bouts.mp4')
        cmd = [
            python_cmd, 'scripts/generate_bouts_video.py',
            '--behavior', args.behavior,
            '--annotations-dir', args.annotations_dir,
            '--video-dir', args.video_dir,
            '--output', all_bouts_video,
            '--workers', str(args.workers)
        ]
        if args.verbose:
            cmd.append('--verbose')
        
        run_command(cmd, "Creating Video of All Bouts", check=False,
                   step_num=current_step, total_steps=total_steps, elapsed_time=elapsed, verbose=args.verbose)
    
    # Step 3: Detect outliers (multiple methods + consensus)
    if not args.skip_outliers:
        current_step += 1
        elapsed = time.time() - pipeline_start_time
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
        
        run_command(cmd, "Detecting Outliers (Multi-Method Consensus)", check=True,
                   step_num=current_step, total_steps=total_steps, elapsed_time=elapsed, verbose=args.verbose)
        
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
                python_cmd, 'scripts/generate_bouts_video.py',
                '--behavior', args.behavior,
                '--annotations-dir', temp_ann_dir,
                '--video-dir', args.video_dir,
                '--output', outlier_video,
                '--workers', str(args.workers)
            ]
            if args.verbose:
                cmd.append('--verbose')
            
            current_step += 1
            elapsed = time.time() - pipeline_start_time
            run_command(cmd, "Creating Video of Outliers", check=False,
                       step_num=current_step, total_steps=total_steps, elapsed_time=elapsed, verbose=args.verbose)
    
    # Step 5: Remove outliers and apply PCA
    if not args.skip_clustering:
        outlier_csv = os.path.join(args.output_dir, 'outliers', 'consensus_outliers.csv')
        
        if os.path.exists(outlier_csv):
            import pandas as pd
            df = pd.read_csv(features_file)
            outliers_df = pd.read_csv(outlier_csv)
            
            # Remove outliers
            df_clean = df[~df['bout_id'].isin(outliers_df['bout_id'])]
            print(f"\n  → Removed {len(outliers_df)} outliers, {len(df_clean)} bouts remaining")
            
            # Save clean features
            features_clean_file = os.path.join(args.output_dir, 'clustering', 'bout_features_clean.csv')
            df_clean.to_csv(features_clean_file, index=False)
            
            # Apply PCA using R
            current_step += 1
            elapsed = time.time() - pipeline_start_time
            pca_features_file = os.path.join(args.output_dir, 'clustering', 'bout_features_pca.csv')
            cmd = [
                'Rscript', '-e',
                f'''
                # Fix R environment
                tryCatch({{ options(editor = "vim") }}, error = function(e) {{}})
                .libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))
                # Explicitly load required packages
                library(utils)
                library(stats)
                df <- read.csv("{features_clean_file}", stringsAsFactors=FALSE)
                metadata_cols <- c("bout_id", "video_name", "animal_id", "start_frame", 
                                 "end_frame", "behavior", "duration_frames")
                feature_cols <- setdiff(colnames(df), metadata_cols)
                X <- as.matrix(df[, feature_cols])
                X[!is.finite(X)] <- 0
                # Remove constant/zero variance columns
                col_vars <- apply(X, 2, stats::var, na.rm=TRUE)
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
            run_command(cmd, "Applying PCA (95% Variance)", check=True,
                       step_num=current_step, total_steps=total_steps, elapsed_time=elapsed, verbose=args.verbose)
        else:
            print("Warning: No outliers file found, using all features")
            pca_features_file = features_file
        
        # Step 6: Cluster with multiple methods
        clustering_methods = ['hierarchical', 'bsoid']
        
        for method in clustering_methods:
            method_name = method.replace('_', ' ').title()
            
            # Clustering
            current_step += 1
            elapsed = time.time() - pipeline_start_time
            cluster_dir = os.path.join(args.output_dir, 'clustering', method)
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Run clustering
            cluster_file = os.path.join(cluster_dir, f'cluster_assignments_{method}.csv')
            
            # B-SOID uses UMAP for dimensionality reduction, so it should use clean features (not PCA)
            # Other methods use PCA features
            input_features_file = features_clean_file if method == 'bsoid' and os.path.exists(features_clean_file) else pca_features_file
            
            cmd = [
                'Rscript', 'BoutAnalysisScripts/scripts/core/cluster_bouts.R',
                '--input', input_features_file,
                '--method', method,
                '--output-dir', cluster_dir,
                '--ncores', str(args.workers)
            ]
            if args.verbose:
                cmd.append('--verbose')
            
            if run_command(cmd, f"Clustering with {method_name}", check=False,
                          step_num=current_step, total_steps=total_steps, elapsed_time=elapsed, verbose=args.verbose):
                if os.path.exists(cluster_file):
                    # Generate PDF visualizations
                    current_step += 1
                    elapsed = time.time() - pipeline_start_time
                    # Note: --method is passed via environment variable due to getopt limitation
                    env = os.environ.copy()
                    env['CLUSTER_METHOD'] = method
                    pdf_cmd = [
                        'Rscript', 'BoutAnalysisScripts/scripts/visualization/visualize_clusters_pdf.R',
                        '--features', input_features_file,  # Use same input file for visualization
                        '--clusters', cluster_file,
                        '--output-dir', cluster_dir
                    ]
                    if args.verbose:
                        pdf_cmd.append('--verbose')
                    
                    # Run with environment variable for method
                    run_command(pdf_cmd, f"Generating PDF Visualizations ({method_name})", check=False, env=env,
                              step_num=current_step, total_steps=total_steps, elapsed_time=elapsed, verbose=args.verbose)
                    
                    # Generate cluster videos
                    current_step += 1
                    elapsed = time.time() - pipeline_start_time
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
                    
                    run_command(video_cmd, f"Generating Cluster Videos ({method_name})", check=False,
                              step_num=current_step, total_steps=total_steps, elapsed_time=elapsed, verbose=args.verbose)
    
    # Step 7: Generate comprehensive analysis report
    if not args.skip_clustering:
        current_step += 1
        elapsed = time.time() - pipeline_start_time
        report_cmd = [
            python_cmd, 'scripts/generate_analysis_report.py',
            '--output-dir', args.output_dir,
            '--behavior', args.behavior
        ]
        if args.workers:
            report_cmd.extend(['--workers', str(args.workers)])
        
        run_command(report_cmd, "Generating Analysis Report", check=False,
                   step_num=current_step, total_steps=total_steps, elapsed_time=elapsed, verbose=args.verbose)
    
    # Summary
    total_duration = time.time() - pipeline_start_time
    print(f"\n{'='*70}")
    print("✓ ANALYSIS PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"Total Duration: {format_time(total_duration)}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll results organized in: {args.output_dir}/")
    print("\nGenerated files:")
    print(f"  ✓ Features: {features_file}")
    if not args.skip_all_bouts_video:
        print(f"  ✓ All bouts video: {args.output_dir}/videos/all_bouts.mp4")
    if not args.skip_outliers:
        print(f"  ✓ Outlier video: {args.output_dir}/videos/outliers.mp4")
        print(f"  ✓ Consensus outliers: {args.output_dir}/outliers/consensus_outliers.csv")
    if not args.skip_clustering:
        print(f"  ✓ PCA features: {args.output_dir}/clustering/bout_features_pca.csv")
        print(f"  ✓ Cluster assignments: {args.output_dir}/clustering/*/cluster_assignments_*.csv")
        print(f"  ✓ PDF reports: {args.output_dir}/clustering/*/clustering_*_report.pdf")
        print(f"  ✓ Cluster videos: {args.output_dir}/videos/*/")
        print(f"  ✓ Analysis report: {args.output_dir}/ANALYSIS_REPORT.md")
    print(f"\nAll output is organized in '{args.output_dir}/' directory for cleanliness.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

