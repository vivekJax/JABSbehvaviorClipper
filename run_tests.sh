#!/bin/bash
# Test runner script for JABS Behavior Clipper

set -e

echo "=========================================="
echo "JABS Behavior Clipper - Test Suite"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Installing test dependencies..."
    pip install -r requirements.txt
fi

# Run tests
echo "Running unit tests..."
pytest tests/ -v --tb=short -m "not slow and not e2e"

echo ""
echo "=========================================="
echo "All unit tests passed!"
echo "=========================================="
echo ""
echo "To run all tests including slow/e2e:"
echo "  pytest tests/ -v"
echo ""
echo "To run with coverage:"
echo "  pytest tests/ --cov=scripts --cov-report=html"
echo ""

