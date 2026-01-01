"""
Unit tests for generate_bouts_video.py
"""
import pytest
import os
import json
from unittest.mock import patch, MagicMock, mock_open

# Import functions to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.generate_bouts_video import (
    validate_video_file,
    validate_frame_range,
    get_video_frame_count,
    get_bouts,
    get_pose_file,
    get_bboxes,
    sec_to_ass_time,
    generate_ass
)


class TestValidateVideoFile:
    """Tests for validate_video_file function."""
    
    def test_validate_video_file_exists(self, temp_dir):
        """Test validation of existing file."""
        video_file = os.path.join(temp_dir, "test.mp4")
        with open(video_file, 'w') as f:
            f.write("dummy")
        
        assert validate_video_file(video_file) is True
    
    def test_validate_video_file_not_exists(self):
        """Test validation of nonexistent file."""
        assert validate_video_file("/nonexistent/file.mp4") is False
    
    def test_validate_video_file_directory(self, temp_dir):
        """Test validation when path is a directory."""
        assert validate_video_file(temp_dir) is False


class TestValidateFrameRange:
    """Tests for validate_frame_range function."""
    
    def test_validate_frame_range_valid(self):
        """Test valid frame range."""
        is_valid, msg = validate_frame_range(100, 200)
        assert is_valid is True
        assert msg == ""
    
    def test_validate_frame_range_negative_start(self):
        """Test negative start frame."""
        is_valid, msg = validate_frame_range(-1, 200)
        assert is_valid is False
        assert "must be >= 0" in msg
    
    def test_validate_frame_range_end_before_start(self):
        """Test end frame before start frame."""
        is_valid, msg = validate_frame_range(200, 100)
        assert is_valid is False
        assert "must be >=" in msg
    
    def test_validate_frame_range_exceeds_video_length(self):
        """Test frame range exceeding video length."""
        is_valid, msg = validate_frame_range(100, 200, num_frames=150)
        assert is_valid is True  # Should clamp
        assert "clamped" in msg or msg == ""


class TestGetVideoFrameCount:
    """Tests for get_video_frame_count function."""
    
    @patch('subprocess.run')
    def test_get_video_frame_count_success(self, mock_run):
        """Test successful frame count extraction."""
        mock_run.return_value = MagicMock(returncode=0, stdout="500\n")
        
        count = get_video_frame_count("test.mp4")
        assert count == 500
    
    @patch('subprocess.run')
    def test_get_video_frame_count_failure(self, mock_run):
        """Test failure to get frame count."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        
        count = get_video_frame_count("test.mp4")
        assert count is None


class TestGetBouts:
    """Tests for get_bouts function."""
    
    def test_get_bouts_basic(self, mock_annotations_dir):
        """Test extracting bouts from annotations."""
        with patch('scripts.generate_bouts_video.ANNOTATIONS_DIR', mock_annotations_dir):
            bouts = get_bouts("turn_left")
            
            assert len(bouts) > 0
            assert all(bout['behavior'] == 'turn_left' for bout in bouts)
            assert all(bout['present'] is True for bout in bouts)
    
    def test_get_bouts_no_annotations(self, temp_dir):
        """Test when no annotation files exist."""
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        with patch('scripts.generate_bouts_video.ANNOTATIONS_DIR', empty_dir):
            bouts = get_bouts("turn_left")
            assert len(bouts) == 0


class TestGetPoseFile:
    """Tests for get_pose_file function."""
    
    def test_get_pose_file_found(self, temp_dir):
        """Test finding pose file."""
        video_name = "test_video.mp4"
        pose_file = os.path.join(temp_dir, "test_video_pose_est_v8.h5")
        with open(pose_file, 'w') as f:
            f.write("dummy")
        
        # get_pose_file in generate_bouts_video.py uses VIDEO_DIR constant
        with patch('scripts.generate_bouts_video.VIDEO_DIR', temp_dir):
            found = get_pose_file(video_name)
            assert found == pose_file
    
    def test_get_pose_file_not_found(self, temp_dir):
        """Test when pose file doesn't exist."""
        with patch('scripts.generate_bouts_video.VIDEO_DIR', temp_dir):
            found = get_pose_file("nonexistent.mp4")
            assert found is None


class TestGetBboxes:
    """Tests for get_bboxes function."""
    
    @patch('subprocess.run')
    def test_get_bboxes_success(self, mock_run, temp_dir):
        """Test successful bounding box extraction."""
        # Mock h5dump output
        mock_output = """0,0,100,200,300,400
0,1,150,250,350,450"""
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)
        
        bboxes = get_bboxes("test.h5", "0", 0, 10)
        
        assert len(bboxes) > 0
        assert all(isinstance(k, int) for k in bboxes.keys())  # Frame numbers as keys
        assert all('x1' in bbox for bbox in bboxes.values())
        assert all('y1' in bbox for bbox in bboxes.values())
    
    @patch('subprocess.run')
    def test_get_bboxes_failure(self, mock_run):
        """Test failure to extract bounding boxes."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        
        bboxes = get_bboxes("test.h5", "0", 0, 10)
        assert len(bboxes) == 0


class TestSecToAssTime:
    """Tests for sec_to_ass_time function."""
    
    def test_sec_to_ass_time_basic(self):
        """Test time conversion."""
        result = sec_to_ass_time(125.5)
        assert result == "0:02:05.50"
    
    def test_sec_to_ass_time_zero(self):
        """Test zero seconds."""
        result = sec_to_ass_time(0.0)
        assert result == "0:00:00.00"
    
    def test_sec_to_ass_time_large(self):
        """Test large time value."""
        result = sec_to_ass_time(3661.5)
        assert "1:01:01" in result


class TestGenerateAss:
    """Tests for generate_ass function."""
    
    def test_generate_ass_basic(self, temp_dir):
        """Test ASS subtitle generation."""
        bboxes = [
            {"frame": 0, "identity": 0, "x1": 100, "y1": 200, "x2": 300, "y2": 400},
            {"frame": 1, "identity": 0, "x1": 110, "y1": 210, "x2": 310, "y2": 410}
        ]
        
        ass_file = os.path.join(temp_dir, "test.ass")
        generate_ass(bboxes, ass_file, fps=30.0)
        
        assert os.path.exists(ass_file)
        with open(ass_file, 'r') as f:
            content = f.read()
            assert "Dialogue" in content
            assert "100,200,300,400" in content

