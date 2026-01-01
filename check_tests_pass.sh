#!/bin/bash
# Script to verify all tests pass
# Usage: ./check_tests_pass.sh

set -e

echo "=========================================="
echo "JABS Behavior Clipper - Test Validation"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "ERROR: pytest is not installed."
    echo ""
    echo "Install test dependencies:"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "Or install pytest directly:"
    echo "  pip install pytest pytest-cov pytest-mock"
    exit 1
fi

# Validate test file syntax first
echo "Step 1: Validating test file syntax..."
python3 validate_tests.py
if [ $? -ne 0 ]; then
    echo "ERROR: Test file validation failed"
    exit 1
fi
echo ""

# Run unit tests (fast)
echo "Step 2: Running unit tests (excluding slow/e2e)..."
if pytest tests/ -v -m "not slow and not e2e" --tb=short; then
    echo ""
    echo "✓ All unit tests passed!"
else
    echo ""
    echo "✗ Some unit tests failed!"
    echo ""
    echo "To see detailed output:"
    echo "  pytest tests/ -v"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test validation complete!"
echo "=========================================="
echo ""
echo "To run all tests including slow/e2e:"
echo "  pytest tests/ -v"
echo ""
echo "To run with coverage:"
echo "  pytest tests/ --cov=scripts --cov-report=html"
echo ""

