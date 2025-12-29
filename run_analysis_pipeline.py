#!/usr/bin/env python3
"""
Unified Analysis Pipeline

Links Python bout/feature extraction with R clustering and outlier detection.
Uses caching to avoid recomputation and save compute time.

Usage:
    python3 run_analysis_pipeline.py --behavior turn_left
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=check)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        if check:
            sys.exit(result.returncode)
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Unified analysis pipeline: Extract features, cluster, and detect outliers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This pipeline runs:
  1. Python: Extract bouts and features from HDF5 files (with caching)
  2. R: Cluster bouts based on features
  3. R: Detect outliers in feature space
  4. R: Visualize results

All steps use caching to avoid recomputation when possible.
        """
    )
    
    parser.add_argument('-b', '--behavior', default='turn_left',
                       help='Behavior name to analyze (default: turn_left)')
    parser.add_argument('-a', '--annotations-dir', default='../jabs/annotations',
                       help='Directory containing annotation JSON files (default: ../jabs/annotations)')
    parser.add_argument('-f', '--features-dir', default=None,
                       help='Base directory for feature HDF5 files (default: auto-detect from ../jabs/features)')
    parser.add_argument('-o', '--output-dir', default='BoutResults',
                       help='Output directory for all results (default: BoutResults)')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature extraction (use existing results/bout_features.csv)')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='Skip clustering analysis')
    parser.add_argument('--skip-outliers', action='store_true',
                       help='Skip outlier detection')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation even if cache exists')
    parser.add_argument('--distance-metric', default='mahalanobis',
                       choices=['euclidean', 'manhattan', 'cosine', 'mahalanobis'],
                       help='Distance metric for clustering/outliers (default: mahalanobis)')
    parser.add_argument('--use-pca', action='store_true', default=True,
                       help='Use PCA reduction for distance calculation (default: True)')
    parser.add_argument('--pca-variance', type=float, default=0.95,
                       help='Proportion of variance to retain in PCA (default: 0.95)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU cores - 1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'cache'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'clustering'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'outlier_videos'), exist_ok=True)
    
    features_file = os.path.join(args.output_dir, 'bout_features.csv')
    
    # Step 1: Extract features (Python)
    if not args.skip_features:
        cmd = [
            'python3', 'scripts/extract_bout_features.py',
            '--behavior', args.behavior,
            '--annotations-dir', args.annotations_dir,
            '--output', features_file,
            '--cache-dir', os.path.join(args.output_dir, 'cache')
        ]
        
        if args.features_dir:
            cmd.extend(['--features-dir', args.features_dir])
        
        if args.workers:
            cmd.extend(['--workers', str(args.workers)])
        
        if args.force_recompute:
            cmd.append('--force-recompute')
        
        if args.verbose:
            cmd.append('--verbose')
        
        if not run_command(cmd, "Step 1: Extracting bout features (Python)", check=True):
            sys.exit(1)
    else:
        if not os.path.exists(features_file):
            print(f"Error: Features file not found: {features_file}")
            print("Run without --skip-features to generate it.")
            sys.exit(1)
        print(f"\nSkipping feature extraction. Using existing file: {features_file}")
    
    # Step 2: Cluster bouts (R)
    if not args.skip_clustering:
        cmd = [
            'Rscript', 'BoutAnalysisScripts/scripts/core/cluster_bouts.R',
            '--features', features_file,
            '--output-dir', os.path.join(args.output_dir, 'clustering'),
            '--method', 'kmeans',
            '--distance-metric', args.distance_metric
        ]
        
        if args.workers:
            cmd.extend(['--ncores', str(args.workers)])
        
        if args.use_pca:
            cmd.append('--use-pca')
            cmd.extend(['--pca-variance', str(args.pca_variance)])
        
        if args.verbose:
            cmd.append('--verbose')
        
        if not run_command(cmd, "Step 2: Clustering bouts (R)", check=False):
            print("Warning: Clustering failed, but continuing...")
    else:
        print("\nSkipping clustering analysis.")
    
    # Step 3: Detect outliers (R)
    if not args.skip_outliers:
        cmd = [
            'Rscript', 'BoutAnalysisScripts/scripts/core/find_outliers.R',
            '--features', features_file,
            '--distance-metric', args.distance_metric,
            '--output-dir', os.path.join(args.output_dir, 'outlier_videos'),
            '--video-dir', '.',
            '--behavior', args.behavior
        ]
        
        if args.use_pca:
            cmd.append('--use-pca')
            cmd.extend(['--pca-variance', str(args.pca_variance)])
        
        if args.verbose:
            cmd.append('--verbose')
        
        if not run_command(cmd, "Step 3: Detecting outliers (R)", check=False):
            print("Warning: Outlier detection failed, but continuing...")
    else:
        print("\nSkipping outlier detection.")
    
    # Step 4: Visualize clusters (R)
    if not args.skip_visualization:
        cluster_file = os.path.join(args.output_dir, 'clustering', 'cluster_assignments_kmeans.csv')
        if os.path.exists(cluster_file):
            cmd = [
                'Rscript', 'BoutAnalysisScripts/scripts/visualization/visualize_clusters.R',
                '--features', features_file,
                '--clusters', cluster_file,
                '--output-dir', os.path.join(args.output_dir, 'clustering')
            ]
            
            if args.verbose:
                cmd.append('--verbose')
            
            if not run_command(cmd, "Step 4: Visualizing clusters (R)", check=False):
                print("Warning: Visualization failed, but continuing...")
        else:
            print(f"\nSkipping visualization: cluster file not found: {cluster_file}")
    else:
        print("\nSkipping visualization.")
    
    # Summary
    print(f"\n{'='*60}")
    print("Analysis Pipeline Complete")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"  - Features: {features_file}")
    
    cluster_file = os.path.join(args.output_dir, 'clustering', 'cluster_assignments_kmeans.csv')
    if os.path.exists(cluster_file):
        print(f"  - Clusters: {cluster_file}")
    
    outlier_file = os.path.join(args.output_dir, 'outlier_videos', 'outliers.csv')
    if os.path.exists(outlier_file):
        print(f"  - Outliers: {outlier_file}")
    
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()

