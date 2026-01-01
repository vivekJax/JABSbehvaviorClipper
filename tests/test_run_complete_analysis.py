"""
Unit tests for run_complete_analysis.py
"""
import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from run_complete_analysis import (
    get_python_cmd,
    format_time,
    print_progress_header,
    run_command
)


class TestGetPythonCmd:
    """Tests for get_python_cmd function."""
    
    @patch('subprocess.run')
    def test_get_python_cmd_system_python(self, mock_run):
        """Test detection of system Python."""
        mock_run.return_value = MagicMock(returncode=0)
        
        cmd = get_python_cmd()
        assert cmd in ['/usr/bin/python3', 'python3']
    
    @patch('subprocess.run')
    def test_get_python_cmd_fallback(self, mock_run):
        """Test fallback to default python3."""
        mock_run.return_value = MagicMock(returncode=1)
        
        cmd = get_python_cmd()
        assert cmd == 'python3'


class TestFormatTime:
    """Tests for format_time function."""
    
    def test_format_time_seconds(self):
        """Test formatting seconds."""
        from run_complete_analysis import format_time
        assert format_time(45.5) == "45.5s"
    
    def test_format_time_minutes(self):
        """Test formatting minutes."""
        from run_complete_analysis import format_time
        assert format_time(125) == "2m 5s"
    
    def test_format_time_hours(self):
        """Test formatting hours."""
        from run_complete_analysis import format_time
        result = format_time(3665)
        assert "h" in result
        assert "m" in result


class TestPrintProgressHeader:
    """Tests for print_progress_header function."""
    
    def test_print_progress_header_basic(self, capsys):
        """Test progress header printing."""
        from run_complete_analysis import print_progress_header
        
        print_progress_header(1, 5, "Test step")
        captured = capsys.readouterr()
        
        assert "Step 1/5" in captured.out
        assert "Test step" in captured.out
        assert "20%" in captured.out
    
    def test_print_progress_header_with_time(self, capsys):
        """Test progress header with elapsed time."""
        from run_complete_analysis import print_progress_header
        
        print_progress_header(2, 4, "Test step", elapsed_time=125.5)
        captured = capsys.readouterr()
        
        assert "Step 2/4" in captured.out
        assert "Elapsed" in captured.out


class TestRunCommand:
    """Tests for run_command function."""
    
    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Success\n", stderr="")
        
        from run_complete_analysis import run_command
        
        result = run_command(['echo', 'test'], "Test command", check=False)
        assert result == 0
    
    @patch('subprocess.run')
    def test_run_command_failure(self, mock_run):
        """Test failed command execution."""
        mock_run.return_value = MagicMock(returncode=1, stdout="Error\n", stderr="")
        
        from run_complete_analysis import run_command
        
        result = run_command(['false'], "Test command", check=False)
        assert result == 1
    
    @patch('subprocess.run')
    def test_run_command_with_progress(self, mock_run, capsys):
        """Test command with progress indication."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Success\n", stderr="")
        
        from run_complete_analysis import run_command
        
        run_command(['echo', 'test'], "Test command", step_num=1, total_steps=3)
        captured = capsys.readouterr()
        
        assert "Step 1/3" in captured.out or "Test command" in captured.out

