# Copyright (c) Meta Platforms, Inc. and affiliates.


TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TESTS_DIR"

echo "======================================"
echo "TextSeal Package Installation Tests"
echo "======================================"
echo ""

# Track pass/fail
TOTAL_TESTS=4
PASSED_TESTS=0
FAILED_TESTS=0

# Test 1: Imports
echo "[1/4] Running import tests..."
python test_imports.py
if [ $? -eq 0 ]; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
echo ""

# Test 2: Use Case 1
echo "[2/4] Running Use Case 1 test (Watermarking + Detection)..."
python test_use_case_1.py
if [ $? -eq 0 ]; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
echo ""

# Test 3: Use Case 2
echo "[3/4] Running Use Case 2 test (Watermarking Only)..."
python test_use_case_2.py
if [ $? -eq 0 ]; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
echo ""

# Test 4: Use Case 3
echo "[4/4] Running Use Case 3 test (Detection Only)..."
python test_use_case_3.py
if [ $? -eq 0 ]; then
    ((PASSED_TESTS++))
else
    ((FAILED_TESTS++))
fi
echo ""

# Summary
echo "======================================"
echo "Test Summary"
echo "======================================"
echo "Total:  $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
