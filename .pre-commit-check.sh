#!/bin/bash
# Pre-commit hook to validate tests pass before committing
# Install: ln -s ../../.pre-commit-check.sh .git/hooks/pre-commit

set -e

echo "Running test validation before commit..."

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "Warning: pytest not installed. Skipping test validation."
    echo "Install with: pip install pytest pytest-cov pytest-mock"
    exit 0
fi

# Run fast tests only (skip slow and e2e)
echo "Running unit tests..."
if pytest tests/ -v -m "not slow and not e2e" --tb=short; then
    echo "✓ All unit tests passed!"
    exit 0
else
    echo "✗ Some tests failed. Please fix before committing."
    echo "Run 'pytest tests/ -v' to see details."
    exit 1
fi

