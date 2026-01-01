"""
Unit tests for plot_bout_trajectories.py
"""
import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.plot_bout_trajectories import (
    get_pose_file,
    get_cage_dimensions,
    get_lixit_location,
    extract_keypoint,
    extract_bbox_centroids,
    extract_nose_keypoints,
    find_correct_identity_index
)


class TestGetPoseFile:
    """Tests for get_pose_file function."""
    
    def test_get_pose_file_found(self, temp_dir, sample_h5_file):
        """Test finding pose file with standard naming."""
        video_name = "test_video.mp4"
        video_dir = os.path.dirname(sample_h5_file)
        
        pose_file = get_pose_file(video_name, video_dir)
        assert pose_file is not None
        assert os.path.exists(pose_file)
    
    def test_get_pose_file_not_found(self, temp_dir):
        """Test when pose file doesn't exist."""
        pose_file = get_pose_file("nonexistent.mp4", temp_dir)
        assert pose_file is None


class TestGetCageDimensions:
    """Tests for get_cage_dimensions function."""
    
    def test_get_cage_dimensions_basic(self, sample_h5_file):
        """Test extracting cage dimensions from HDF5 file."""
        max_x, max_y = get_cage_dimensions(sample_h5_file)
        
        assert max_x is not None
        assert max_y is not None
        assert max_x > 0
        assert max_y > 0
    
    def test_get_cage_dimensions_nonexistent_file(self):
        """Test handling of nonexistent file."""
        max_x, max_y = get_cage_dimensions("/nonexistent/file.h5")
        assert max_x is None
        assert max_y is None


class TestGetLixitLocation:
    """Tests for get_lixit_location function."""
    
    def test_get_lixit_location_basic(self, sample_h5_file, temp_dir):
        """Test extracting lixit location."""
        video_name = "test_video.mp4"
        video_dir = os.path.dirname(sample_h5_file)
        
        lixit = get_lixit_location(video_name, video_dir)
        
        # Should return tuple of (x, y) or None
        if lixit is not None:
            assert len(lixit) == 2
            assert lixit[0] > 0  # x coordinate
            assert lixit[1] > 0  # y coordinate
    
    def test_get_lixit_location_not_found(self, temp_dir):
        """Test when lixit location not found."""
        lixit = get_lixit_location("nonexistent.mp4", temp_dir)
        assert lixit is None


class TestExtractKeypoint:
    """Tests for extract_keypoint function."""
    
    def test_extract_keypoint_basic(self, sample_h5_file):
        """Test extracting keypoint positions."""
        keypoints = extract_keypoint(sample_h5_file, 0, 100, 150, 0)
        
        assert len(keypoints) > 0
        assert all(len(kp) == 3 for kp in keypoints)  # (frame, x, y)
        assert all(kp[0] >= 100 and kp[0] <= 150 for kp in keypoints)
    
    def test_extract_keypoint_invalid_range(self, sample_h5_file):
        """Test handling of invalid frame range."""
        keypoints = extract_keypoint(sample_h5_file, 0, 1000, 2000, 0)
        # Should return empty list or handle gracefully
        assert isinstance(keypoints, list)


class TestExtractBboxCentroids:
    """Tests for extract_bbox_centroids function."""
    
    def test_extract_bbox_centroids_basic(self, sample_h5_file):
        """Test extracting bounding box centroids."""
        centroids = extract_bbox_centroids(sample_h5_file, 0, 100, 150)
        
        assert len(centroids) > 0
        assert all(len(c) == 3 for c in centroids)  # (frame, x, y)
        assert all(c[0] >= 100 and c[0] <= 150 for c in centroids)
    
    def test_extract_bbox_centroids_filters_invalid(self, sample_h5_file):
        """Test that invalid bounding boxes are filtered out."""
        centroids = extract_bbox_centroids(sample_h5_file, 0, 100, 150)
        
        # All centroids should have valid coordinates
        for frame, x, y in centroids:
            assert x > 0
            assert y > 0
            assert np.isfinite(x)
            assert np.isfinite(y)


class TestExtractNoseKeypoints:
    """Tests for extract_nose_keypoints function."""
    
    def test_extract_nose_keypoints_basic(self, sample_h5_file):
        """Test extracting nose keypoints."""
        nose_points = extract_nose_keypoints(sample_h5_file, 0, 100, 150)
        
        assert len(nose_points) > 0
        assert all(len(p) == 3 for p in nose_points)  # (frame, x, y)


class TestFindCorrectIdentityIndex:
    """Tests for find_correct_identity_index function."""
    
    def test_find_correct_identity_index_basic(self, sample_h5_file):
        """Test finding correct identity index."""
        identity_idx = find_correct_identity_index(sample_h5_file, "0", 100, 150)
        
        # Should return integer index or None
        assert isinstance(identity_idx, (int, type(None)))
        if identity_idx is not None:
            assert identity_idx >= 0

