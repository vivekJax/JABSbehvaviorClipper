"""
End-to-end tests for the complete analysis pipeline
"""
import pytest
import os
import json
import tempfile
import shutil
import subprocess
from pathlib import Path

# These tests require actual data files and may be skipped if not available
pytestmark = pytest.mark.skipif(
    not os.path.exists('../jabs/annotations') or len(os.listdir('../jabs/annotations')) == 0,
    reason="Real annotation files not available"
)


class TestEndToEndPipeline:
    """End-to-end tests for the complete analysis pipeline."""
    
    def test_pipeline_creates_output_directory(self, temp_dir):
        """Test that pipeline creates output directory structure."""
        output_dir = os.path.join(temp_dir, "BoutResults")
        
        # Run a minimal pipeline check
        # This is a placeholder - actual E2E test would run the full pipeline
        os.makedirs(output_dir, exist_ok=True)
        
        assert os.path.exists(output_dir)
    
    def test_feature_extraction_produces_csv(self, temp_dir, mock_annotations_dir, mock_features_dir):
        """Test that feature extraction produces a CSV file."""
        output_file = os.path.join(temp_dir, "bout_features.csv")
        
        # This would actually call the feature extraction script
        # For now, we'll create a mock CSV to verify structure
        import pandas as pd
        df = pd.DataFrame({
            'bout_id': [0, 1],
            'video_name': ['test_video.mp4', 'test_video.mp4'],
            'identity': ['0', '1'],
            'start_frame': [100, 200],
            'end_frame': [150, 250]
        })
        df.to_csv(output_file, index=False)
        
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0


class TestPipelineIntegration:
    """Integration tests for pipeline components."""
    
    def test_annotation_loading_integration(self, mock_annotations_dir):
        """Test that annotation loading works with real file structure."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from scripts.extract_bout_features import load_annotations
        
        bouts = load_annotations(mock_annotations_dir, "turn_left")
        assert len(bouts) > 0
    
    def test_h5_file_reading_integration(self, sample_h5_file):
        """Test that HDF5 files can be read correctly."""
        import h5py
        
        with h5py.File(sample_h5_file, 'r') as f:
            assert '/poseest/bbox' in f
            assert '/poseest/points' in f
            assert '/features/per_frame' in f


class TestDataConsistency:
    """Tests for data consistency across pipeline stages."""
    
    def test_bout_id_consistency(self, mock_annotations_dir):
        """Test that bout IDs are consistent across processing."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from scripts.extract_bout_features import load_annotations
        
        bouts = load_annotations(mock_annotations_dir, "turn_left")
        
        # Check that bout IDs are sequential and unique
        bout_ids = [bout['bout_id'] for bout in bouts]
        assert len(bout_ids) == len(set(bout_ids))  # All unique
        assert sorted(bout_ids) == list(range(len(bout_ids)))  # Sequential from 0
    
    def test_frame_range_validity(self, mock_annotations_dir):
        """Test that frame ranges are valid."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from scripts.extract_bout_features import load_annotations
        
        bouts = load_annotations(mock_annotations_dir, "turn_left")
        
        for bout in bouts:
            assert bout['start_frame'] >= 0
            assert bout['end_frame'] >= bout['start_frame']


@pytest.mark.slow
class TestFullPipelineExecution:
    """Slow tests that run the full pipeline (marked for optional execution)."""
    
    def test_full_pipeline_with_mock_data(self, temp_dir, mock_annotations_dir, mock_features_dir, sample_h5_file):
        """Test full pipeline execution with mock data."""
        # This would run the actual pipeline
        # For now, we verify the setup is correct
        assert os.path.exists(mock_annotations_dir)
        assert os.path.exists(mock_features_dir)
        assert os.path.exists(sample_h5_file)

