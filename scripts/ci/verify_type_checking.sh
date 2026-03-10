#!/bin/bash
# ============================================================================
# Type Checking Implementation Verification Script
# ============================================================================
# This script validates the complete PEP 561 type checking setup for Python
# projects using the UV template.
#
# TEMPLATE USAGE:
# ---------------
# This is a generic verification script for Python UV template projects.
# Child projects should customize the configuration below for their specific
# package structure and type definitions.
#
# QUICK START FOR CHILD PROJECTS:
# --------------------------------
# 1. Set your package name:
#    PACKAGE_NAME=your_package ./scripts/ci/verify_type_checking.sh
#
# 2. Set your documentation directory (child repos typically use 'docs'):
#    DOCS_DIR=docs ./scripts/ci/verify_type_checking.sh
#
# 3. Or edit the defaults in the configuration section below
#
# 4. (Optional) Add expected type definitions to EXPECTED_TYPES array
#
# CUSTOMIZATION OPTIONS:
# ----------------------
# - PACKAGE_NAME: Your package directory name (default: 'src')
# - DOCS_DIR: Documentation directory name (default: 'docs_with_docusarus')
#              Child repos typically use: 'docs'
# - EXPECTED_TYPES: Array of type definitions to verify in types.py
#
# WHAT THIS SCRIPT CHECKS:
# ------------------------
# ✅ py.typed marker file exists
# ✅ types.py module with __all__ exports
# ✅ pyproject.toml includes py.typed in artifacts
# ✅ Package exports types in __init__.py
# ✅ Type checking workflow exists
# ✅ MyPy passes on all package files
# ✅ Type imports work correctly
# ✅ Documentation exists
# ✅ Examples pass type checking
# ✅ Package distribution includes type information
#
# EXAMPLES:
# ---------
# # Use defaults (template repo: src + docs_with_docusarus)
# ./scripts/ci/verify_type_checking.sh
#
# # Child project with standard structure (package: my_package, docs: docs)
# PACKAGE_NAME=my_package DOCS_DIR=docs ./scripts/ci/verify_type_checking.sh
#
# # Specify custom package name only
# PACKAGE_NAME=my_package ./scripts/ci/verify_type_checking.sh
#
# # Verify specific types exist with custom docs directory
# EXPECTED_TYPES=("MyType" "MyProtocol") PACKAGE_NAME=my_pkg DOCS_DIR=docs ./scripts/ci/verify_type_checking.sh
# ============================================================================

set -e  # Exit on error

# === CONFIGURATION (Customize for your project) ===
# Default package name - override by setting environment variable:
# PACKAGE_NAME=your_package ./scripts/ci/verify_type_checking.sh
PACKAGE_NAME="${PACKAGE_NAME:-src}"

# Documentation directory name - override by setting environment variable:
# DOCS_DIR=docs ./scripts/ci/verify_type_checking.sh
# Template repo uses: docs_with_docusarus
# Child repos typically use: docs
DOCS_DIR="${DOCS_DIR:-docs_with_docusarus}"

# Expected type definitions in types.py (customize for your project)
# Example: EXPECTED_TYPES=("MyEventPayload" "HandlerProtocol" "ConfigDict")
EXPECTED_TYPES=("")

echo "=========================================="
echo "Type Checking Implementation Verification"
echo "=========================================="
echo "Package: ${PACKAGE_NAME}"
echo "Documentation: ${DOCS_DIR}"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

# Helper function for test results
pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

# Test 1: Check py.typed marker file
echo "Test 1: Checking py.typed marker file..."
if [ -f "${PACKAGE_NAME}/py.typed" ]; then
    pass "py.typed marker file exists at ${PACKAGE_NAME}/py.typed"
else
    fail "py.typed marker file not found at ${PACKAGE_NAME}/py.typed"
    warn "Create an empty file at ${PACKAGE_NAME}/py.typed to mark your package as typed"
fi
echo ""

# Test 2: Check types.py module
echo "Test 2: Checking types.py module..."
if [ -f "${PACKAGE_NAME}/types.py" ]; then
    pass "types.py module exists at ${PACKAGE_NAME}/types.py"

    # Check for __all__ export
    if grep -q "__all__" "${PACKAGE_NAME}/types.py"; then
        pass "types.py has __all__ export"
    else
        warn "types.py missing __all__ export (recommended for explicit API)"
    fi

    # Check for key type definitions (customize for your project)
    if [ ${#EXPECTED_TYPES[@]} -gt 0 ] && [ -n "${EXPECTED_TYPES[0]}" ]; then
        for type_name in "${EXPECTED_TYPES[@]}"; do
            if grep -q "$type_name" "${PACKAGE_NAME}/types.py"; then
                pass "types.py contains $type_name"
            else
                warn "types.py missing $type_name (expected type definition)"
            fi
        done
    else
        warn "No expected types configured. Set EXPECTED_TYPES in script or skip this check."
    fi
else
    fail "types.py module not found at ${PACKAGE_NAME}/types.py"
    warn "Create ${PACKAGE_NAME}/types.py to define your project's type aliases and protocols"
fi
echo ""

# Test 3: Check pyproject.toml configuration
echo "Test 3: Checking pyproject.toml configuration..."
if grep -q "py.typed" pyproject.toml; then
    pass "pyproject.toml includes py.typed in artifacts"
else
    fail "pyproject.toml missing py.typed in artifacts"
fi
echo ""

# Test 4: Check __init__.py exports
echo "Test 4: Checking package exports..."
if [ -f "${PACKAGE_NAME}/__init__.py" ]; then
    # Check if types module is exported
    if grep -q "types" "${PACKAGE_NAME}/__init__.py"; then
        pass "__init__.py exports types module"
    else
        warn "__init__.py missing types export (recommended for type checking)"
    fi
else
    fail "${PACKAGE_NAME}/__init__.py not found"
fi
echo ""

# Test 5: Check CI workflow
echo "Test 5: Checking CI workflow..."
if [ -f ".github/workflows/type-check.yml" ]; then
    pass "Type checking workflow exists"

    # Validate YAML syntax
    if command -v python3 &> /dev/null; then
        if python3 -c "import yaml; yaml.safe_load(open('.github/workflows/type-check.yml'))" 2>/dev/null; then
            pass "Workflow YAML syntax is valid"
        else
            fail "Workflow YAML syntax is invalid"
        fi
    else
        warn "Python3 not available, skipping YAML validation"
    fi
else
    fail "Type checking workflow not found"
fi
echo ""

# Test 6: Run MyPy type checking
echo "Test 6: Running MyPy type checking..."
if command -v uv &> /dev/null; then
    echo "Checking types.py..."
    if [ -f "${PACKAGE_NAME}/types.py" ]; then
        if uv run mypy "${PACKAGE_NAME}/types.py" --show-error-codes 2>&1 | grep -q "Success"; then
            pass "types.py passes MyPy"
        else
            fail "types.py has MyPy errors"
        fi
    else
        warn "Skipping types.py check (file not found)"
    fi

    echo "Checking __init__.py..."
    if [ -f "${PACKAGE_NAME}/__init__.py" ]; then
        if uv run mypy "${PACKAGE_NAME}/__init__.py" --show-error-codes 2>&1 | grep -q "Success"; then
            pass "__init__.py passes MyPy"
        else
            fail "__init__.py has MyPy errors"
        fi
    else
        warn "Skipping __init__.py check (file not found)"
    fi

    # Check entire package
    echo "Checking entire package..."
    if uv run mypy "${PACKAGE_NAME}" --show-error-codes 2>&1 | grep -q "Success"; then
        pass "Package passes MyPy type checking"
    else
        warn "Package has MyPy errors (review above output)"
    fi
else
    warn "uv not available, skipping MyPy checks"
fi
echo ""

# Test 7: Test type imports
echo "Test 7: Testing type imports..."
if command -v uv &> /dev/null; then
    # Test basic package import
    if uv run python -c "import ${PACKAGE_NAME}" 2>/dev/null; then
        pass "Package imports successfully"
    else
        fail "Failed to import package"
    fi

    # Test types module import if it exists
    if [ -f "${PACKAGE_NAME}/types.py" ]; then
        if uv run python -c "from ${PACKAGE_NAME} import types" 2>/dev/null; then
            pass "Types module imports successfully"
        else
            warn "Failed to import types module (check __init__.py exports)"
        fi

        # Test __all__ export if types module exists
        if uv run python -c "from ${PACKAGE_NAME} import types; print(getattr(types, '__all__', []))" 2>/dev/null; then
            pass "Types module __all__ attribute accessible"
        else
            warn "Types module missing __all__ or has import errors"
        fi
    else
        warn "Skipping types import test (types.py not found)"
    fi
else
    warn "uv not available, skipping import tests"
fi
echo ""

# Test 8: Check documentation
echo "Test 8: Checking documentation..."
DOCS_FOUND=0

# Check for Docusaurus documentation (common path structures)
if [ -f "${DOCS_DIR}/contents/development/type-checking.mdx" ]; then
    pass "Type checking documentation exists at ${DOCS_DIR}/contents/development/type-checking.mdx"
    ((DOCS_FOUND++))
elif [ -f "${DOCS_DIR}/development/type-checking.mdx" ]; then
    pass "Type checking documentation exists at ${DOCS_DIR}/development/type-checking.mdx"
    ((DOCS_FOUND++))
elif [ -f "${DOCS_DIR}/type-checking.mdx" ] || [ -f "${DOCS_DIR}/type-checking.md" ]; then
    pass "Type checking documentation exists in ${DOCS_DIR}/"
    ((DOCS_FOUND++))
fi

# Check for workflow documentation
if [ -f "${DOCS_DIR}/contents/development/ci-cd/type-checking-workflow.mdx" ]; then
    pass "Workflow documentation exists at ${DOCS_DIR}/contents/development/ci-cd/"
    ((DOCS_FOUND++))
elif [ -f "${DOCS_DIR}/development/ci-cd/type-checking-workflow.mdx" ]; then
    pass "Workflow documentation exists at ${DOCS_DIR}/development/ci-cd/"
    ((DOCS_FOUND++))
elif [ -f "${DOCS_DIR}/ci-cd/type-checking-workflow.mdx" ] || [ -f "${DOCS_DIR}/workflows/type-checking.md" ]; then
    pass "Workflow documentation exists in ${DOCS_DIR}/"
    ((DOCS_FOUND++))
fi

# Check for quick reference guide (root level)
if [ -f "TYPE_CHECKING_GUIDE.md" ]; then
    pass "Quick reference guide exists (TYPE_CHECKING_GUIDE.md)"
    ((DOCS_FOUND++))
fi

if [ $DOCS_FOUND -eq 0 ]; then
    warn "No type checking documentation found in ${DOCS_DIR}/ (recommended to add)"
else
    pass "Found $DOCS_FOUND documentation file(s)"
fi
echo ""

# Test 9: Check examples
echo "Test 9: Checking examples..."
if [ -f "examples/type_checking/type_checking_example.py" ]; then
    pass "Type checking example exists"

    if command -v uv &> /dev/null; then
        if uv run mypy examples/type_checking/type_checking_example.py 2>&1 | grep -q "Success"; then
            pass "Example passes MyPy"
        else
            fail "Example has MyPy errors"
        fi
    fi
else
    fail "Type checking example not found"
fi

if [ -f "examples/type_checking/README.md" ]; then
    pass "Examples README exists"
else
    fail "Examples README not found"
fi
echo ""

# Test 10: Build package and verify distribution
echo "Test 10: Verifying package distribution..."
if command -v uv &> /dev/null; then
    echo "Building package..."
    if uv build --sdist --wheel > /dev/null 2>&1; then
        pass "Package builds successfully"

        # Check sdist for py.typed
        if tar -tzf dist/*.tar.gz 2>/dev/null | grep -q "${PACKAGE_NAME}/py.typed"; then
            pass "py.typed included in source distribution"
        else
            warn "py.typed not in source distribution (check pyproject.toml artifacts)"
        fi

        # Check sdist for types.py
        if tar -tzf dist/*.tar.gz 2>/dev/null | grep -q "${PACKAGE_NAME}/types.py"; then
            pass "types.py included in source distribution"
        else
            warn "types.py not in source distribution (only needed if you have types.py)"
        fi

        # Check wheel for py.typed
        if unzip -l dist/*.whl 2>/dev/null | grep -q "${PACKAGE_NAME}/py.typed"; then
            pass "py.typed included in wheel"
        else
            warn "py.typed not in wheel (check pyproject.toml artifacts)"
        fi

        # Check wheel for types.py
        if unzip -l dist/*.whl 2>/dev/null | grep -q "${PACKAGE_NAME}/types.py"; then
            pass "types.py included in wheel"
        else
            warn "types.py not in wheel (only needed if you have types.py)"
        fi

        # Cleanup
        rm -rf dist/ build/ 2>/dev/null
    else
        fail "Package build failed"
    fi
else
    warn "uv not available, skipping package build test"
fi
echo ""

# Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! Type checking implementation is complete.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Customize type definitions in ${PACKAGE_NAME}/types.py"
    echo "  2. Add type checking examples in examples/type_checking/"
    echo "  3. Update documentation with project-specific type information"
    echo "  4. Run: uv run mypy ${PACKAGE_NAME} to verify all types"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Please review the errors above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  - Create ${PACKAGE_NAME}/py.typed to mark package as typed"
    echo "  - Add type stubs in ${PACKAGE_NAME}/types.py"
    echo "  - Export types in ${PACKAGE_NAME}/__init__.py"
    echo "  - Add 'py.typed' to pyproject.toml artifacts section"
    echo "  - Run: uv run mypy ${PACKAGE_NAME} --install-types"
    exit 1
fi
