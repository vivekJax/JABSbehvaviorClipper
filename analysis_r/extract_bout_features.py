#!/usr/bin/env python3
# This script will use the virtual environment if available
# To use venv: source analysis_r/venv/bin/activate
# To use conda: conda activate behavior_analysis
"""
Extract and aggregate features for behavior bouts from HDF5 files

This script:
1. Loads annotation JSON files to get bout frame ranges
2. Matches bouts to HDF5 feature files
3. Extracts per-frame features for each bout
4. Aggregates features to bout-level statistics (mean, std, min, max, etc.)
5. Outputs CSV file for R analysis

Usage:
    python3 analysis_r/extract_bout_features.py --behavior turn_left --output bout_features.csv
"""

import json
import argparse
import glob
import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def load_annotations(annotations_dir, behavior_name):
    """Load all annotation files and extract bouts for specified behavior."""
    json_files = glob.glob(os.path.join(annotations_dir, '*.json'))
    
    print(f"Loading annotations from {annotations_dir}")
    print(f"Found {len(json_files)} annotation files")
    
    all_bouts = []
    bout_counter = 0
    
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            video_name = data.get('file')
            if not video_name:
                continue
            
            labels = data.get('labels', {})
            
            for identity_id, behaviors in labels.items():
                behavior_bouts = behaviors.get(behavior_name, [])
                
                if isinstance(behavior_bouts, list):
                    for bout in behavior_bouts:
                        if bout.get('present') is True:
                            bout_counter += 1
                            all_bouts.append({
                                'bout_id': bout_counter,
                                'video_name': video_name,
                                'identity': str(identity_id),
                                'start_frame': bout.get('start'),
                                'end_frame': bout.get('end'),
                                'behavior': behavior_name
                            })
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue
    
    return all_bouts


def get_feature_file_path(video_name, animal_id, features_dir):
    """Get path to feature HDF5 file for a video and animal."""
    video_basename = os.path.splitext(video_name)[0]
    feature_path = os.path.join(features_dir, video_basename, animal_id, 'features.h5')
    
    if os.path.exists(feature_path):
        return feature_path
    return None


def extract_bout_features(h5_file_path, start_frame, end_frame, verbose=False):
    """Extract all per-frame features for a bout from HDF5 file."""
    features = {}
    
    try:
        if not os.path.exists(h5_file_path):
            if verbose:
                print(f"  File does not exist: {h5_file_path}")
            return features
        
        with h5py.File(h5_file_path, 'r') as f:
            # Navigate to /features/per_frame group
            if '/features/per_frame' not in f:
                if verbose:
                    print(f"  /features/per_frame group not found")
                return features
            
            per_frame_group = f['/features/per_frame']
            
            # Recursively find all datasets
            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Get relative path from /features/per_frame
                    rel_path = name.replace('/features/per_frame/', '')
                    if rel_path:
                        # Replace / with _ for feature name
                        feature_name = rel_path.replace('/', '_')
                        try:
                            # Read entire dataset
                            data = obj[:]
                            # Handle multi-dimensional arrays
                            if len(data.shape) > 1:
                                data = data[:, 0] if data.shape[1] > 0 else data.flatten()
                            
                            # Extract frame range (0-indexed to 1-indexed)
                            total_frames = len(data)
                            start_idx = max(0, min(start_frame, total_frames - 1))
                            end_idx = max(start_idx, min(end_frame, total_frames - 1))
                            
                            if start_idx <= end_idx:
                                bout_data = data[start_idx:end_idx + 1]
                                features[feature_name] = bout_data
                        except Exception as e:
                            if verbose:
                                print(f"    Error reading {name}: {e}")
            
            per_frame_group.visititems(find_datasets)
            
    except Exception as e:
        if verbose:
            print(f"  Error extracting features: {e}")
    
    return features


def aggregate_bout_features(per_frame_features):
    """Aggregate per-frame features to bout-level statistics."""
    aggregated = {}
    
    for feature_name, values in per_frame_features.items():
        if len(values) == 0:
            continue
        
        # Convert to numpy array for easier computation
        values = np.array(values, dtype=float)
        
        # Skip if all NaN
        if np.all(np.isnan(values)):
            continue
        
        # Compute statistics
        aggregated[f"{feature_name}_mean"] = np.nanmean(values)
        aggregated[f"{feature_name}_std"] = np.nanstd(values)
        aggregated[f"{feature_name}_min"] = np.nanmin(values)
        aggregated[f"{feature_name}_max"] = np.nanmax(values)
        aggregated[f"{feature_name}_median"] = np.nanmedian(values)
        aggregated[f"{feature_name}_first"] = values[0] if len(values) > 0 else np.nan
        aggregated[f"{feature_name}_last"] = values[-1] if len(values) > 0 else np.nan
        aggregated[f"{feature_name}_duration"] = len(values)
        
        # Calculate IQR if we have enough values
        if len(values) >= 4:
            q25, q75 = np.nanpercentile(values, [25, 75])
            aggregated[f"{feature_name}_iqr"] = q75 - q25
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Extract bout features from HDF5 files')
    parser.add_argument('-b', '--behavior', default='turn_left',
                       help='Behavior name to extract (default: turn_left)')
    parser.add_argument('-a', '--annotations-dir', default='jabs/annotations',
                       help='Directory containing annotation JSON files')
    parser.add_argument('-f', '--features-dir', default=None,
                       help='Base directory for feature HDF5 files (default: auto-detect)')
    parser.add_argument('-o', '--output', default='results/bout_features.csv',
                       help='Output CSV file (default: results/bout_features.csv)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Auto-detect features directory if not specified
    if args.features_dir is None:
        possible_paths = [
            'jabs/features',
            '../jabs/features',
            '../../jabs/features',
            './jabs/features'
        ]
        
        for path in possible_paths:
            if os.path.isdir(path):
                args.features_dir = os.path.abspath(path)
                if args.verbose:
                    print(f"Auto-detected features directory: {args.features_dir}")
                break
        
        if args.features_dir is None:
            args.features_dir = 'jabs/features'
            if args.verbose:
                print(f"Using default features directory: {args.features_dir}")
    
    # Load annotations
    bouts = load_annotations(args.annotations_dir, args.behavior)
    
    print(f"Found {len(bouts)} bouts for behavior '{args.behavior}'")
    
    if len(bouts) == 0:
        print(f"Error: No bouts found for behavior '{args.behavior}'")
        sys.exit(1)
    
    # Extract features
    print(f"\nExtracting features for {len(bouts)} bouts of behavior '{args.behavior}'")
    print(f"Features directory: {args.features_dir}")
    
    results = []
    missing_files_count = 0
    missing_files_examples = []
    
    for i, bout in enumerate(bouts):
        if (i + 1) % 10 == 0:
            print(f"Processing bout {i + 1}/{len(bouts)}")
        
        # Match bout to feature file
        feature_file = get_feature_file_path(bout['video_name'], bout['identity'], args.features_dir)
        
        if feature_file is None:
            missing_files_count += 1
            if len(missing_files_examples) < 3:
                video_basename = os.path.splitext(bout['video_name'])[0]
                expected_path = os.path.join(args.features_dir, video_basename, bout['identity'], 'features.h5')
                missing_files_examples.append({
                    'video': bout['video_name'],
                    'animal': bout['identity'],
                    'expected': expected_path
                })
            if args.verbose:
                print(f"No feature file found for bout: {bout['video_name']} animal {bout['identity']}")
            continue
        
        # Extract per-frame features
        per_frame_features = extract_bout_features(
            feature_file,
            bout['start_frame'],
            bout['end_frame'],
            verbose=args.verbose and (i < 3)
        )
        
        if len(per_frame_features) == 0:
            if args.verbose:
                print(f"No features extracted for bout {i + 1}: {bout['video_name']} animal {bout['identity']}")
            continue
        
        # Aggregate features
        aggregated_features = aggregate_bout_features(per_frame_features)
        
        # Combine with bout metadata
        result_row = {
            'bout_id': bout['bout_id'],
            'video_name': bout['video_name'],
            'animal_id': bout['identity'],
            'start_frame': bout['start_frame'],
            'end_frame': bout['end_frame'],
            'behavior': bout['behavior']
        }
        result_row.update(aggregated_features)
        
        results.append(result_row)
    
    print(f"\nSuccessfully extracted features for {len(results)}/{len(bouts)} bouts")
    
    if missing_files_count > 0:
        print(f"\nWarning: {missing_files_count} bouts had no feature files found")
        if len(missing_files_examples) > 0 and args.verbose:
            print("Example missing feature file paths:")
            for example in missing_files_examples:
                print(f"  Video: {example['video']}, Animal: {example['animal']}")
                print(f"    Expected: {example['expected']}")
    
    if len(results) == 0:
        print("Error: No features extracted. Check feature file paths and frame ranges.")
        sys.exit(1)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    
    # Ensure columns are in a consistent order: metadata first, then features
    metadata_cols = ['bout_id', 'video_name', 'animal_id', 'start_frame', 'end_frame', 'behavior']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(feature_cols)]
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print(f"Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"\nOutput file: {args.output}")
    print(f"\nFile Contents:")
    print(f"  - Total bouts: {len(df)}")
    print(f"  - Metadata columns: {len(metadata_cols)}")
    print(f"    * bout_id: Unique identifier for each bout")
    print(f"    * video_name: Source video file")
    print(f"    * animal_id: Animal identifier")
    print(f"    * start_frame: Starting frame of bout")
    print(f"    * end_frame: Ending frame of bout")
    print(f"    * behavior: Behavior label")
    print(f"  - Feature columns: {len(feature_cols)}")
    print(f"    * Each feature has aggregated statistics:")
    print(f"      - <feature>_mean: Mean value during bout")
    print(f"      - <feature>_std: Standard deviation")
    print(f"      - <feature>_min: Minimum value")
    print(f"      - <feature>_max: Maximum value")
    print(f"      - <feature>_median: Median value")
    print(f"      - <feature>_first: First frame value")
    print(f"      - <feature>_last: Last frame value")
    print(f"      - <feature>_duration: Bout duration (frames)")
    print(f"      - <feature>_iqr: Interquartile range")
    print(f"\n  Total columns: {len(df.columns)}")
    print(f"\nThis file is used for:")
    print(f"  - Outlier detection (find_outliers.R)")
    print(f"  - Clustering analysis (cluster_bouts.R)")
    print(f"  - All downstream analysis")
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()

