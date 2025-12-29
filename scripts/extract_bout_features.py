#!/usr/bin/env python3
"""
Extract and aggregate features for behavior bouts from HDF5 files

This script:
1. Loads annotation JSON files to get bout frame ranges (using unfragmented_labels)
2. Matches bouts to HDF5 feature files
3. Extracts per-frame features for each bout
4. Aggregates features to bout-level statistics (mean, std, min, max, etc.)
5. Outputs CSV file for R analysis
6. Supports caching to avoid recomputation

Usage:
    python3 extract_bout_features.py --behavior turn_left --output results/bout_features.csv
"""

import json
import argparse
import glob
import os
import sys
import h5py
import numpy as np
import pandas as pd
import hashlib
import pickle
import multiprocessing
from pathlib import Path
from datetime import datetime
from functools import partial


def compute_cache_key(annotations_dir, behavior_name, features_dir):
    """Compute a cache key based on annotation files and behavior."""
    json_files = sorted(glob.glob(os.path.join(annotations_dir, '*.json')))
    
    # Hash file modification times and paths
    hash_input = f"{behavior_name}:{features_dir}:"
    for json_file in json_files:
        if os.path.exists(json_file):
            mtime = os.path.getmtime(json_file)
            hash_input += f"{json_file}:{mtime}:"
    
    return hashlib.md5(hash_input.encode()).hexdigest()


def load_annotations(annotations_dir, behavior_name):
    """Load all annotation files and extract bouts for specified behavior.
    Uses unfragmented_labels to match GUI counts."""
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
            
            # Use unfragmented_labels to match GUI counts (original bout boundaries)
            # unfragmented_labels contains the original start/end frames as labeled
            # labels contains fragmented bouts (broken up to exclude frames missing pose)
            labels = data.get('unfragmented_labels', {})
            
            # Fall back to labels if unfragmented_labels doesn't exist
            if not labels:
                labels = data.get('labels', {})
                print(f"Warning: unfragmented_labels not found in {os.path.basename(json_path)}, using labels instead")
            
            for identity_id, behaviors in labels.items():
                behavior_bouts = behaviors.get(behavior_name, [])
                
                if isinstance(behavior_bouts, list):
                    for bout in behavior_bouts:
                        # Only include present=True bouts for feature extraction
                        if bout.get('present') is True:
                            bout_counter += 1
                            all_bouts.append({
                                'bout_id': bout_counter - 1,  # 0-indexed for consistency with R
                                'video_name': video_name,
                                'identity': str(identity_id),
                                'start_frame': bout.get('start'),
                                'end_frame': bout.get('end'),
                                'behavior': behavior_name,
                                'present': True
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
    
    # Try alternative paths
    alternatives = [
        feature_path,
        os.path.join(features_dir, video_basename, f"{animal_id}.h5"),
        os.path.join(features_dir, video_name, animal_id, 'features.h5'),
    ]
    
    for alt_path in alternatives:
        if os.path.exists(alt_path):
            return alt_path
    
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


def process_bout_worker(bout_data):
    """Worker function for parallel bout processing.
    
    Args:
        bout_data: Tuple of (bout_index, bout_dict, features_dir, verbose)
    
    Returns:
        Tuple of (bout_index, result_dict) or (bout_index, None) if failed
    """
    bout_index, bout, features_dir, verbose = bout_data
    
    try:
        # Match bout to feature file
        feature_file = get_feature_file_path(bout['video_name'], bout['identity'], features_dir)
        
        if feature_file is None:
            return (bout_index, None)
        
        # Extract per-frame features
        per_frame_features = extract_bout_features(
            feature_file,
            bout['start_frame'],
            bout['end_frame'],
            verbose=verbose
        )
        
        if len(per_frame_features) == 0:
            return (bout_index, None)
        
        # Aggregate features
        aggregated_features = aggregate_bout_features(per_frame_features)
        
        # Combine with bout metadata
        result_row = {
            'bout_id': bout['bout_id'],
            'video_name': bout['video_name'],
            'animal_id': bout['identity'],
            'start_frame': bout['start_frame'],
            'end_frame': bout['end_frame'],
            'behavior': bout['behavior'],
            'duration_frames': bout['end_frame'] - bout['start_frame'] + 1
        }
        result_row.update(aggregated_features)
        
        return (bout_index, result_row)
    except Exception as e:
        if verbose:
            print(f"Error processing bout {bout_index}: {e}")
        return (bout_index, None)


def main():
    parser = argparse.ArgumentParser(description='Extract bout features from HDF5 files')
    parser.add_argument('-b', '--behavior', default='turn_left',
                       help='Behavior name to extract (default: turn_left)')
    parser.add_argument('-a', '--annotations-dir', default='../jabs/annotations',
                       help='Directory containing annotation JSON files (default: ../jabs/annotations)')
    parser.add_argument('-f', '--features-dir', default=None,
                       help='Base directory for feature HDF5 files (default: auto-detect from ../jabs/features)')
    parser.add_argument('-o', '--output', default='BoutResults/bout_features.csv',
                       help='Output CSV file (default: BoutResults/bout_features.csv)')
    parser.add_argument('--cache-dir', default='BoutResults/cache',
                       help='Directory for caching intermediate results (default: BoutResults/cache)')
    parser.add_argument('--use-cache', action='store_true', default=True,
                       help='Use cached results if available (default: True)')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation even if cache exists')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU cores - 1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Auto-detect features directory if not specified
    if args.features_dir is None:
        possible_paths = [
            '../jabs/features',  # Parent directory (most common)
            '../../jabs/features',
            'jabs/features',
            './jabs/features'
        ]
        
        for path in possible_paths:
            if os.path.isdir(path):
                args.features_dir = os.path.abspath(path)
                if args.verbose:
                    print(f"Auto-detected features directory: {args.features_dir}")
                break
        
        if args.features_dir is None:
            args.features_dir = '../jabs/features'
            if args.verbose:
                print(f"Using default features directory: {args.features_dir}")
    
    # Check cache
    cache_key = compute_cache_key(args.annotations_dir, args.behavior, args.features_dir)
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f'bout_features_{cache_key}.pkl')
    cache_info_file = os.path.join(cache_dir, f'bout_features_{cache_key}.info')
    
    # Try to load from cache
    if args.use_cache and not args.force_recompute and os.path.exists(cache_file):
        try:
            print(f"Loading cached results from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify cache is still valid
            if os.path.exists(cache_info_file):
                with open(cache_info_file, 'r') as f:
                    cache_info = json.load(f)
                    print(f"Cache created: {cache_info.get('timestamp', 'unknown')}")
            
            df = cached_data['dataframe']
            
            # Save to output location
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            df.to_csv(args.output, index=False)
            
            print(f"\n{'='*60}")
            print(f"Feature Extraction Complete (from cache)")
            print(f"{'='*60}")
            print(f"\nOutput file: {args.output}")
            print(f"  - Total bouts: {len(df)}")
            print(f"  - Total columns: {len(df.columns)}")
            print(f"\n{'='*60}")
            return
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Recomputing...")
    
    # Load annotations
    bouts = load_annotations(args.annotations_dir, args.behavior)
    
    print(f"\nFound {len(bouts)} bouts for behavior '{args.behavior}' (present=True only)")
    
    if len(bouts) == 0:
        print(f"Error: No bouts found for behavior '{args.behavior}'")
        sys.exit(1)
    
    # Determine number of workers (default: n-1 cores)
    if args.workers is None:
        default_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = default_workers
    else:
        num_workers = max(1, args.workers)
    
    # Extract features
    print(f"\nExtracting features for {len(bouts)} bouts of behavior '{args.behavior}'")
    print(f"Features directory: {args.features_dir}")
    print(f"Using {num_workers} parallel workers")
    
    results = []
    missing_files_count = 0
    missing_files_examples = []
    
    # Prepare data for parallel processing
    use_parallel = num_workers > 1 and len(bouts) > 1
    
    if use_parallel:
        # Prepare arguments for parallel processing
        bout_args = [(i, bout, args.features_dir, args.verbose and (i < 3)) 
                     for i, bout in enumerate(bouts)]
        
        # Process bouts in parallel
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use imap_unordered for progress tracking
            completed = 0
            results_dict = {}
            for result in pool.imap_unordered(process_bout_worker, bout_args):
                completed += 1
                if completed % 10 == 0 or completed == len(bouts):
                    print(f"Processed {completed}/{len(bouts)} bouts")
                
                bout_index, result_row = result
                if result_row is None:
                    # Track missing files
                    bout = bouts[bout_index]
                    missing_files_count += 1
                    if len(missing_files_examples) < 3:
                        video_basename = os.path.splitext(bout['video_name'])[0]
                        expected_path = os.path.join(args.features_dir, video_basename, 
                                                    bout['identity'], 'features.h5')
                        missing_files_examples.append({
                            'video': bout['video_name'],
                            'animal': bout['identity'],
                            'expected': expected_path
                        })
                else:
                    results_dict[bout_index] = result_row
        
        # Convert to list and sort by bout_id to maintain order
        results = [results_dict[i] for i in sorted(results_dict.keys())]
    else:
        # Sequential processing
        for i, bout in enumerate(bouts):
            if (i + 1) % 10 == 0:
                print(f"Processing bout {i + 1}/{len(bouts)}")
            
            # Match bout to feature file
            feature_file = get_feature_file_path(bout['video_name'], bout['identity'], args.features_dir)
            
            if feature_file is None:
                missing_files_count += 1
                if len(missing_files_examples) < 3:
                    video_basename = os.path.splitext(bout['video_name'])[0]
                    expected_path = os.path.join(args.features_dir, video_basename, 
                                             bout['identity'], 'features.h5')
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
                'behavior': bout['behavior'],
                'duration_frames': bout['end_frame'] - bout['start_frame'] + 1
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
    metadata_cols = ['bout_id', 'video_name', 'animal_id', 'start_frame', 'end_frame', 'behavior', 'duration_frames']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(feature_cols)]
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    # Save to cache
    if args.use_cache:
        try:
            cache_data = {
                'dataframe': df,
                'bouts': bouts,
                'cache_key': cache_key,
                'timestamp': datetime.now().isoformat()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            cache_info = {
                'cache_key': cache_key,
                'timestamp': datetime.now().isoformat(),
                'behavior': args.behavior,
                'annotations_dir': args.annotations_dir,
                'features_dir': args.features_dir,
                'n_bouts': len(bouts),
                'n_results': len(results)
            }
            with open(cache_info_file, 'w') as f:
                json.dump(cache_info, f, indent=2)
            
            print(f"\nCached results to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print(f"Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"\nOutput file: {args.output}")
    print(f"\nFile Contents:")
    print(f"  - Total bouts: {len(df)}")
    print(f"  - Metadata columns: {len(metadata_cols)}")
    print(f"  - Feature columns: {len(feature_cols)}")
    print(f"  - Total columns: {len(df.columns)}")
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()

