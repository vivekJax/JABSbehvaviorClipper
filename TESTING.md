# Testing Guide

This document describes how to run the test suite for the Behavior Video Generator.

## Test Structure

The test suite is organized in the `tests/` directory:

```
tests/
├── __init__.py
├── test_validation.py      # Tests for validation functions
├── test_data_extraction.py # Tests for data extraction functions
└── test_video_processing.py # Tests for video processing functions
```

## Prerequisites

### Required
- Python 3.6 or higher
- Standard library only (no external packages required for basic tests)

### Optional (for full test coverage)
- `pytest` (recommended for better test output): `pip install pytest`
- `ffmpeg` and `ffprobe` (for integration tests)
- `h5dump` (for bounding box extraction tests)

## Running Tests

### Method 1: Using unittest (Built-in)

Run all tests:
```bash
python3 -m unittest discover tests -v
```

Run a specific test file:
```bash
python3 -m unittest tests.test_validation -v
python3 -m unittest tests.test_data_extraction -v
python3 -m unittest tests.test_video_processing -v
```

Run a specific test class:
```bash
python3 -m unittest tests.test_validation.TestValidation -v
```

Run a specific test method:
```bash
python3 -m unittest tests.test_validation.TestValidation.test_validate_frame_range_valid -v
```

### Method 2: Using pytest (Recommended)

Install pytest:
```bash
pip install pytest
```

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pip install pytest-cov
pytest tests/ --cov=generate_bouts_video --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_validation.py -v
```

Run specific test:
```bash
pytest tests/test_validation.py::TestValidation::test_validate_frame_range_valid -v
```

### Method 3: Direct Execution

Run test files directly:
```bash
python3 tests/test_validation.py
python3 tests/test_data_extraction.py
python3 tests/test_video_processing.py
```

## Test Categories

### Unit Tests

**test_validation.py**
- `validate_video_file()`: Tests file existence and type validation
- `validate_frame_range()`: Tests frame range validation logic
- `get_video_frame_count()`: Tests video metadata extraction

**test_data_extraction.py**
- `get_bouts()`: Tests behavior bout extraction from annotations
- `get_pose_file()`: Tests pose file name generation
- `get_bboxes()`: Tests bounding box extraction from HDF5 files (requires h5dump)
  - Validates coordinate extraction and parsing
  - Tests handling of missing pose files
  - Verifies bounding box data structure

**test_video_processing.py**
- `sec_to_ass_time()`: Tests time format conversion
- `generate_ass()`: Tests ASS subtitle file generation

### Integration Tests

Some tests require real files to be present:
- Tests in `TestDataExtractionIntegration` use actual annotation files if available
- These tests are automatically skipped if files don't exist

## Test Data

### Mock Data
Most tests use temporary files and mock data created during test execution.

### Real Data (Optional)
Some integration tests can use real annotation files from `jabs/annotations/` if available:
- Tests will automatically skip if files are not found
- No modification of real data occurs during testing

## Expected Output

### Successful Test Run

```
test_validate_frame_range_valid (tests.test_validation.TestValidation) ... ok
test_validate_video_file_exists (tests.test_validation.TestValidation) ... ok
test_get_bouts_finds_present_bouts (tests.test_data_extraction.TestDataExtraction) ... ok
...

----------------------------------------------------------------------
Ran X tests in Y.YYYs

OK
```

### Skipped Tests

```
test_get_video_frame_count_existing (tests.test_validation.TestValidation) ... skipped 'Test video file not found'
```

### Failed Tests

```
test_validate_frame_range_negative_start (tests.test_validation.TestValidation) ... FAIL

======================================================================
FAIL: test_validate_frame_range_negative_start (tests.test_validation.TestValidation)
----------------------------------------------------------------------
Traceback (most recent call last):
  ...
AssertionError: ...
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install pytest
      - name: Run tests
        run: pytest tests/ -v
```

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test

```python
import unittest
from generate_bouts_video import my_function

class TestMyFunction(unittest.TestCase):
    def test_my_function_basic(self):
        """Test basic functionality."""
        result = my_function("input")
        self.assertEqual(result, "expected_output")
    
    def test_my_function_edge_case(self):
        """Test edge case."""
        result = my_function("")
        self.assertIsNone(result)
```

### Best Practices
1. **Isolation**: Each test should be independent
2. **Cleanup**: Use `setUp()` and `tearDown()` for fixtures
3. **Descriptive names**: Test names should describe what they test
4. **Docstrings**: Add docstrings to test methods
5. **Assertions**: Use specific assertions (`assertEqual`, `assertIn`, etc.)

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`, ensure you're running tests from the project root:
```bash
cd /path/to/small_test_project
python3 -m unittest discover tests -v
```

### Permission Errors
Some tests create temporary files. Ensure you have write permissions in the test directory.

### Missing Dependencies
Tests that require external tools (ffmpeg, h5dump) will be skipped if tools are not available. This is expected behavior.

### Test Failures
1. Check error messages for specific assertion failures
2. Run tests with `-v` (verbose) for more details
3. Use `--pdb` with pytest to debug: `pytest tests/ --pdb`

## Coverage Goals

Aim for:
- **Unit tests**: >80% code coverage
- **Integration tests**: Cover main workflows
- **Edge cases**: Test error conditions and boundary values

## Additional Resources

- [Python unittest documentation](https://docs.python.org/3/library/unittest.html)
- [pytest documentation](https://docs.pytest.org/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)

