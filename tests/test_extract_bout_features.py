"""
Unit tests for extract_bout_features.py
"""
import pytest
import os
import json
import tempfile
import shutil
import h5py
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.extract_bout_features import (
    compute_cache_key,
    load_annotations,
    get_feature_file_path,
    extract_bout_features,
    aggregate_bout_features
)


class TestComputeCacheKey:
    """Tests for compute_cache_key function."""
    
    def test_cache_key_consistency(self, temp_dir, sample_annotation_data):
        """Test that cache key is consistent for same inputs."""
        annotations_dir = os.path.join(temp_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        annotation_file = os.path.join(annotations_dir, "test.json")
        with open(annotation_file, 'w') as f:
            json.dump(sample_annotation_data, f)
        
        key1 = compute_cache_key(annotations_dir, "turn_left", "/features")
        key2 = compute_cache_key(annotations_dir, "turn_left", "/features")
        
        assert key1 == key2
    
    def test_cache_key_different_behavior(self, temp_dir, sample_annotation_data):
        """Test that cache key differs for different behaviors."""
        annotations_dir = os.path.join(temp_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        annotation_file = os.path.join(annotations_dir, "test.json")
        with open(annotation_file, 'w') as f:
            json.dump(sample_annotation_data, f)
        
        key1 = compute_cache_key(annotations_dir, "turn_left", "/features")
        key2 = compute_cache_key(annotations_dir, "grooming", "/features")
        
        assert key1 != key2


class TestLoadAnnotations:
    """Tests for load_annotations function."""
    
    def test_load_annotations_basic(self, mock_annotations_dir, sample_annotation_data):
        """Test loading annotations with unfragmented_labels."""
        bouts = load_annotations(mock_annotations_dir, "turn_left")
        
        assert len(bouts) == 3  # 2 from identity 0, 1 from identity 1
        assert all(bout['behavior'] == 'turn_left' for bout in bouts)
        assert all(bout['present'] is True for bout in bouts)
        assert bouts[0]['bout_id'] == 0
        assert bouts[0]['start_frame'] == 100
        assert bouts[0]['end_frame'] == 150
    
    def test_load_annotations_fallback_to_labels(self, temp_dir):
        """Test fallback to labels when unfragmented_labels missing."""
        annotations_dir = os.path.join(temp_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Create annotation without unfragmented_labels
        annotation_data = {
            "file": "test_video.mp4",
            "labels": {
                "0": {
                    "turn_left": [
                        {"start": 100, "end": 150, "present": True}
                    ]
                }
            }
        }
        
        annotation_file = os.path.join(annotations_dir, "test.json")
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        bouts = load_annotations(annotations_dir, "turn_left")
        assert len(bouts) == 1
    
    def test_load_annotations_filters_present_false(self, temp_dir):
        """Test that present=False bouts are filtered out."""
        annotations_dir = os.path.join(temp_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        annotation_data = {
            "file": "test_video.mp4",
            "unfragmented_labels": {
                "0": {
                    "turn_left": [
                        {"start": 100, "end": 150, "present": True},
                        {"start": 200, "end": 250, "present": False}
                    ]
                }
            }
        }
        
        annotation_file = os.path.join(annotations_dir, "test.json")
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        bouts = load_annotations(annotations_dir, "turn_left")
        assert len(bouts) == 1
        assert bouts[0]['start_frame'] == 100


class TestGetFeatureFilePath:
    """Tests for get_feature_file_path function."""
    
    def test_get_feature_file_path_standard(self, mock_features_dir):
        """Test finding feature file with standard path."""
        path = get_feature_file_path("test_video.mp4", "0", mock_features_dir)
        assert path is not None
        assert os.path.exists(path)
        assert "test_video" in path
        assert "0" in path
        assert path.endswith("features.h5")
    
    def test_get_feature_file_path_not_found(self, temp_dir):
        """Test when feature file doesn't exist."""
        path = get_feature_file_path("nonexistent.mp4", "0", temp_dir)
        assert path is None


class TestExtractBoutFeatures:
    """Tests for extract_bout_features function."""
    
    def test_extract_bout_features_basic(self, sample_h5_file):
        """Test extracting features from HDF5 file."""
        features = extract_bout_features(sample_h5_file, 100, 150, verbose=False)
        
        assert len(features) > 0
        assert 'velocity_x' in features or any('velocity' in k for k in features.keys())
    
    def test_extract_bout_features_nonexistent_file(self):
        """Test handling of nonexistent file."""
        features = extract_bout_features("/nonexistent/file.h5", 100, 150)
        assert len(features) == 0
    
    def test_extract_bout_features_invalid_frame_range(self, sample_h5_file):
        """Test handling of invalid frame range."""
        features = extract_bout_features(sample_h5_file, 1000, 2000)
        # Should handle gracefully, may return empty or partial features
        assert isinstance(features, dict)


class TestAggregateBoutFeatures:
    """Tests for aggregate_bout_features function."""
    
    def test_aggregate_bout_features_basic(self):
        """Test feature aggregation."""
        per_frame_features = {
            'velocity_x': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'velocity_y': np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
            'speed': np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        }
        
        aggregated = aggregate_bout_features(per_frame_features)
        
        assert 'velocity_x_mean' in aggregated
        assert 'velocity_x_std' in aggregated
        assert 'velocity_x_min' in aggregated
        assert 'velocity_x_max' in aggregated
        assert aggregated['velocity_x_mean'] == pytest.approx(3.0)
    
    def test_aggregate_bout_features_empty(self):
        """Test aggregation with empty features."""
        aggregated = aggregate_bout_features({})
        assert len(aggregated) == 0

