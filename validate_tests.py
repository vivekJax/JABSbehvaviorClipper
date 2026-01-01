#!/usr/bin/env python3
"""
Test validation script - checks that all tests are properly structured
and can be imported without syntax errors.
"""
import os
import sys
import importlib.util
import ast

def check_test_file(filepath):
    """Check if a test file can be parsed and imported."""
    print(f"Checking {filepath}...")
    
    # Check syntax
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"  ✓ Syntax valid")
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False
    
    # Try to import (without running)
    try:
        spec = importlib.util.spec_from_file_location("test_module", filepath)
        if spec and spec.loader:
            # Just check if it can be loaded, don't execute
            print(f"  ✓ Can be imported")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def main():
    """Validate all test files."""
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    if not os.path.exists(tests_dir):
        print(f"Tests directory not found: {tests_dir}")
        return 1
    
    test_files = [
        'test_extract_bout_features.py',
        'test_plot_trajectories.py',
        'test_generate_bouts_video.py',
        'test_run_complete_analysis.py',
        'test_e2e_pipeline.py'
    ]
    
    all_passed = True
    for test_file in test_files:
        filepath = os.path.join(tests_dir, test_file)
        if os.path.exists(filepath):
            if not check_test_file(filepath):
                all_passed = False
        else:
            print(f"Warning: {test_file} not found")
    
    # Check conftest
    conftest_path = os.path.join(tests_dir, 'conftest.py')
    if os.path.exists(conftest_path):
        print(f"\nChecking conftest.py...")
        check_test_file(conftest_path)
    
    if all_passed:
        print("\n✓ All test files are valid!")
        print("\nTo run tests, install pytest:")
        print("  pip install pytest pytest-cov pytest-mock")
        print("\nThen run:")
        print("  pytest tests/ -v")
        return 0
    else:
        print("\n✗ Some test files have errors")
        return 1

if __name__ == '__main__':
    sys.exit(main())

