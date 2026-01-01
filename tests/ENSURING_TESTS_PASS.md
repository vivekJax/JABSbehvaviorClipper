# Ensuring Tests Pass When Making Changes

This guide explains how to ensure all tests pass when you modify the codebase.

## Quick Checklist

Before committing changes:

1. ✅ Run `./check_tests_pass.sh` to validate tests
2. ✅ Fix any failing tests
3. ✅ Ensure all tests pass before pushing

## Test Validation Tools

### 1. `validate_tests.py` - Syntax Checker

**Purpose**: Validates that all test files have correct Python syntax and can be imported.

**Usage**:
```bash
python3 validate_tests.py
```

**What it checks**:
- Test files parse correctly (no syntax errors)
- Test files can be imported as modules
- All required test files exist

**When to run**: Before running tests, or if you've modified test files.

### 2. `check_tests_pass.sh` - Full Test Validation

**Purpose**: Comprehensive test validation that ensures all unit tests pass.

**Usage**:
```bash
./check_tests_pass.sh
```

**What it does**:
1. Validates test file syntax
2. Checks pytest is installed
3. Runs all unit tests (excluding slow/e2e)
4. Reports pass/fail status

**When to run**: 
- Before committing changes
- After modifying any code
- Before creating pull requests

### 3. `.pre-commit-check.sh` - Git Pre-commit Hook

**Purpose**: Automatically runs tests before allowing commits.

**Setup**:
```bash
ln -s ../../.pre-commit-check.sh .git/hooks/pre-commit
```

**What it does**: Runs fast unit tests before each commit. Prevents committing if tests fail.

**When it runs**: Automatically on every `git commit`.

## Running Tests Manually

### Run All Unit Tests (Fast)
```bash
pytest tests/ -v -m "not slow and not e2e"
```

### Run Specific Test File
```bash
pytest tests/test_extract_bout_features.py -v
```

### Run Specific Test
```bash
pytest tests/test_extract_bout_features.py::TestLoadAnnotations::test_load_annotations_basic -v
```

### Run with Coverage
```bash
pytest tests/ --cov=scripts --cov-report=html -m "not slow and not e2e"
```

### Run All Tests (Including Slow/E2E)
```bash
pytest tests/ -v
```

## Understanding Test Failures

### Common Failure Types

1. **Import Errors**
   - **Symptom**: `ModuleNotFoundError` or `ImportError`
   - **Fix**: Check that you're running from project root, install dependencies

2. **Assertion Failures**
   - **Symptom**: `AssertionError` with specific condition
   - **Fix**: Your code change broke expected behavior. Review the test to understand what it expects.

3. **Fixture Errors**
   - **Symptom**: `FixtureNotFoundError`
   - **Fix**: Check that `conftest.py` fixtures are available

4. **Mock Errors**
   - **Symptom**: Mock not called or called incorrectly
   - **Fix**: Update mocks to match new function signatures

### Debugging Failed Tests

1. **Run with verbose output**:
   ```bash
   pytest tests/test_xyz.py -vv
   ```

2. **Run with print statements**:
   ```bash
   pytest tests/test_xyz.py -s
   ```

3. **Run specific failing test**:
   ```bash
   pytest tests/test_xyz.py::TestClass::test_method -vv
   ```

4. **Use Python debugger**:
   ```python
   import pdb; pdb.set_trace()  # Add to test
   ```

## When Tests Should Be Updated

### Update Tests When:

1. **Function signature changes**
   - Update test calls to match new parameters
   - Update mocks to match new behavior

2. **New functionality added**
   - Add new tests for new functions
   - Add tests for new edge cases

3. **Behavior changes intentionally**
   - Update test expectations to match new behavior
   - Update test documentation

4. **Bug fixes**
   - Add regression tests to prevent bug from returning
   - Update existing tests if bug fix changes expected behavior

### Don't Update Tests When:

1. **Tests reveal bugs**
   - Fix the code, not the test
   - Tests are correct, code is wrong

2. **Tests are "too strict"**
   - Tests should be strict to catch regressions
   - If test is wrong, fix it properly with explanation

## CI/CD Integration

Tests automatically run in GitHub Actions on:
- Every push to `main` or `develop`
- Every pull request

**CI will fail if**:
- Test syntax is invalid
- Any unit test fails
- Test coverage drops significantly

**To see CI results**:
1. Go to GitHub repository
2. Click "Actions" tab
3. View latest workflow run

## Test Maintenance

### Adding New Tests

1. Create test in appropriate test file
2. Follow naming convention: `test_<function_name>_<scenario>`
3. Add docstring explaining what/why
4. Use fixtures from `conftest.py` when possible
5. Run tests to ensure they pass

### Test Best Practices

1. **One assertion per test** (when possible)
2. **Clear test names** that describe what's being tested
3. **Use fixtures** for common setup
4. **Test edge cases** (empty inputs, None values, etc.)
5. **Test error handling** (file not found, invalid inputs, etc.)
6. **Document why** each test matters

## Troubleshooting

### "pytest not found"
```bash
pip install -r requirements.txt
```

### "Module not found" errors
```bash
# Ensure you're in project root
cd /path/to/JABSbehvaviorClipper
pytest tests/
```

### "Fixture not found"
- Check that `conftest.py` exists and has the fixture
- Ensure fixture name matches exactly

### Tests pass locally but fail in CI
- Check Python version (CI uses multiple versions)
- Check dependencies are in `requirements.txt`
- Check for platform-specific code

## Summary

**Before every commit**:
```bash
./check_tests_pass.sh
```

**If tests fail**:
1. Read the error message
2. Understand what the test expects
3. Fix your code to match expectations
4. Re-run tests
5. Commit only when all tests pass

**Remember**: Tests are your safety net. They catch bugs before they reach production!

