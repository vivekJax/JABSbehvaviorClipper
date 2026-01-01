# Test Suite for JABS Behavior Clipper

Comprehensive unit tests and end-to-end tests for the behavior analysis pipeline.

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Pytest fixtures and configuration
├── test_extract_bout_features.py  # Tests for feature extraction
├── test_plot_trajectories.py      # Tests for trajectory plotting
├── test_generate_bouts_video.py   # Tests for video generation
├── test_run_complete_analysis.py  # Tests for pipeline orchestration
└── test_e2e_pipeline.py          # End-to-end integration tests
```

## Test Explanations

For detailed explanations of what each test does and why it matters, see:
- **[TEST_EXPLANATIONS.md](TEST_EXPLANATIONS.md)** - Comprehensive guide to all tests

Each test includes:
- **WHAT IT DOES**: Description of the test's behavior
- **WHY IT MATTERS**: Explanation of why this test is important for code quality

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
# Using pytest (recommended)
pytest tests/ -v

# Using unittest
python -m pytest tests/ -v
```

### Run Specific Test Files

```bash
# Unit tests for feature extraction
pytest tests/test_extract_bout_features.py -v

# Unit tests for trajectory plotting
pytest tests/test_plot_trajectories.py -v

# Unit tests for video generation
pytest tests/test_generate_bouts_video.py -v

# End-to-end tests
pytest tests/test_e2e_pipeline.py -v
```

### Run Specific Test Classes

```bash
pytest tests/test_extract_bout_features.py::TestLoadAnnotations -v
```

### Run Specific Test Functions

```bash
pytest tests/test_extract_bout_features.py::TestLoadAnnotations::test_load_annotations_basic -v
```

### Run with Coverage

```bash
# Install coverage tool
pip install pytest-cov

# Run tests with coverage report
pytest tests/ --cov=scripts --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

### Skip Slow Tests

```bash
# Skip tests marked as 'slow'
pytest tests/ -v -m "not slow"
```

## Test Categories

### Unit Tests

**test_extract_bout_features.py**
- `compute_cache_key()`: Cache key generation and consistency
- `load_annotations()`: Annotation file loading and parsing
- `get_feature_file_path()`: Feature file path resolution
- `extract_bout_features()`: HDF5 feature extraction
- `aggregate_bout_features()`: Feature aggregation statistics

**test_plot_trajectories.py**
- `get_pose_file()`: Pose file discovery
- `get_cage_dimensions()`: Cage dimension extraction
- `get_lixit_location()`: Lixit location extraction
- `extract_keypoint()`: Keypoint extraction from HDF5
- `extract_bbox_centroids()`: Bounding box centroid calculation
- `extract_nose_keypoints()`: Nose keypoint extraction

**test_generate_bouts_video.py**
- `validate_video_file()`: Video file validation
- `validate_frame_range()`: Frame range validation
- `get_video_frame_count()`: Video metadata extraction
- `get_bouts()`: Bout extraction from annotations
- `get_pose_file()`: Pose file discovery
- `get_bboxes()`: Bounding box extraction
- `sec_to_ass_time()`: Time format conversion
- `generate_ass()`: ASS subtitle generation

**test_run_complete_analysis.py**
- `get_python_cmd()`: Python interpreter detection
- `format_time()`: Time formatting utilities
- `print_progress_header()`: Progress display
- `run_command()`: Command execution wrapper

### End-to-End Tests

**test_e2e_pipeline.py**
- Full pipeline execution with mock data
- Data consistency across pipeline stages
- Integration between components
- Output file generation verification

## Test Fixtures

Test fixtures are defined in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `sample_annotation_data`: Sample annotation JSON structure
- `sample_annotation_file`: Sample annotation file on disk
- `sample_h5_file`: Sample HDF5 file with pose data
- `sample_cluster_assignments`: Sample cluster assignment DataFrame
- `mock_video_dir`: Mock video directory structure
- `mock_annotations_dir`: Mock annotations directory
- `mock_features_dir`: Mock features directory

## Writing New Tests

### Example Unit Test

```python
def test_my_function_basic():
    """Test basic functionality of my_function."""
    result = my_function(input_data)
    assert result == expected_output
```

### Example Test with Fixtures

```python
def test_load_data_with_fixture(sample_annotation_file):
    """Test loading data using fixture."""
    data = load_annotations(sample_annotation_file)
    assert len(data) > 0
```

### Example Test with Mocking

```python
@patch('subprocess.run')
def test_external_command(mock_run):
    """Test external command execution."""
    mock_run.return_value = MagicMock(returncode=0, stdout="Success")
    result = run_external_command()
    assert result == "Success"
```

## Test Markers

Tests can be marked for selective execution:

- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.e2e`: End-to-end tests

## Continuous Integration

Tests are designed to run in CI/CD pipelines. For GitHub Actions, add:

```yaml
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ -v --cov=scripts --cov-report=xml
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/JABSbehvaviorClipper
pytest tests/
```

### Missing Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

### HDF5 File Issues

Some tests require HDF5 files. If tests fail, check that `h5py` is installed:
```bash
pip install h5py
```

### Permission Errors

Ensure test directories are writable. Tests use temporary directories by default.

## Coverage Goals

- **Unit Tests**: >80% code coverage
- **Integration Tests**: Cover all major workflows
- **End-to-End Tests**: Verify complete pipeline execution

## Contributing

When adding new features:

1. Write unit tests for new functions
2. Add integration tests for new workflows
3. Update this README if adding new test categories
4. Ensure all tests pass before submitting PR

