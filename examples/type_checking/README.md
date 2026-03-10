# Type Checking Examples

> **Template Repository**: This is a template directory. Child projects should implement their own type checking examples demonstrating project-specific type features.

This directory provides a foundation for implementing type checking examples in your Python project.

## 📋 For Child Projects: Implementing Your Examples

When using this template, replace the placeholder `type_checking_example.py` with real examples that demonstrate your project's type system.

### Recommended Example Content

Create examples that showcase:

#### 1. **Core Type Definitions**
```python
# Example: Demonstrate your project's primary types
from your_package import types

# Show Protocol types
class YourHandlerProtocol(types.HandlerProtocol):
    """Example handler following your protocol."""
    pass

# Show type aliases
config: types.ConfigDict = {...}
```

#### 2. **Type Guards and Validation**
```python
# Example: Show runtime type validation
from your_package.types import is_valid_identifier

identifier = "example_123"
if types.is_valid_identifier(identifier):
    print(f"Valid identifier: {identifier}")
```

#### 3. **Generic Types and TypeVars**
```python
# Example: Demonstrate generic type usage
from typing import TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
```

#### 4. **Protocol and Structural Subtyping**
```python
# Example: Show Protocol usage
from typing import Protocol

class Processable(Protocol):
    def process(self) -> None: ...

class DataProcessor:
    def process(self) -> None:
        print("Processing...")
```

## 🚀 Running Your Examples

### Type Check the Example

```bash
# Run MyPy on the example
uv run mypy examples/type_checking/type_checking_example.py

# Type check with strict mode
uv run mypy examples/type_checking/type_checking_example.py --strict
```

### Execute the Example

```bash
# Run the example code
uv run python examples/type_checking/type_checking_example.py
```

### Verify Type Exports

```bash
# Test that types can be imported
uv run python -c "from your_package import types; print(types.__all__)"
```

### Run Type Checking Verification Script

The template includes a comprehensive verification script that checks your entire type checking setup:

```bash
# For template repository (default configuration)
./scripts/ci/verify_type_checking.sh

# For child repositories with standard structure
PACKAGE_NAME=your_package DOCS_DIR=docs ./scripts/ci/verify_type_checking.sh

# Or edit the script defaults at lines 66 and 65:
# PACKAGE_NAME="${PACKAGE_NAME:-your_package}"
# DOCS_DIR="${DOCS_DIR:-docs}"
```

The script verifies:
- ✅ `py.typed` marker file exists
- ✅ `types.py` module with proper exports
- ✅ Package configuration in `pyproject.toml`
- ✅ Type checking workflow exists
- ✅ MyPy passes on all files
- ✅ Documentation exists
- ✅ Examples pass type checking
- ✅ Package distribution includes type information

## 📝 Example Template Structure

Your `type_checking_example.py` should follow this structure:

```python
"""
Example demonstrating type checking with the [Your Project] package.

This example shows how to use the types module for static type checking
with MyPy and other type checkers.

Run MyPy on this file to verify type checking:
    uv run mypy examples/type_checking/type_checking_example.py
"""

from __future__ import annotations

from typing import Any
from your_package import types

# Example 1: Using basic type annotations
def example_basic_types() -> None:
    """Demonstrate basic type usage."""
    pass

# Example 2: Using Protocol types
def example_protocols() -> None:
    """Demonstrate Protocol-based structural typing."""
    pass

# Example 3: Using type guards
def example_type_guards() -> None:
    """Demonstrate runtime type validation."""
    pass

# Main demonstration
def main() -> None:
    """Run all type checking examples."""
    print("=== Type Checking Examples ===\n")
    
    example_basic_types()
    example_protocols()
    example_type_guards()
    
    print("\n✓ All type checking examples completed successfully!")

if __name__ == "__main__":
    main()
```

## 📖 Expected Output Format

Your example should produce clear, structured output:

```
=== Type Checking Examples ===

Example 1: Basic Types
✓ Type annotations working correctly

Example 2: Protocol Types  
✓ Protocol compliance verified

Example 3: Type Guards
✓ Runtime validation successful

✓ All type checking examples completed successfully!
```

## 🔗 Related Documentation

Update these links to point to your project's documentation:

**For Template Repository (`docs_with_docusarus/`):**
- [Type Checking with MyPy](../../docs_with_docusarus/contents/development/type-checking.mdx)
- [Type Checking Workflow](../../docs_with_docusarus/contents/development/ci-cd/type-checking-workflow.mdx)

**For Child Repositories (typically `docs/`):**
- Type Checking with MyPy: `../../docs/development/type-checking.md`
- Type Checking Workflow: `../../docs/development/ci-cd/type-checking-workflow.md`

**Common Documentation:**
- [Quick Reference Guide](../../TYPE_CHECKING_GUIDE.md)
- [Types Module README](../../src/types.README.md) *(if you create one)*

> **Note**: Child projects should update these paths to match their documentation structure.

## ✅ Implementation Checklist

When implementing your type checking examples:

- [ ] Create meaningful examples that demonstrate your project's type system
- [ ] Add comprehensive docstrings explaining each example
- [ ] Include type annotations for all functions and variables
- [ ] Ensure all examples pass `uv run mypy` without errors
- [ ] Test examples can be executed: `uv run python examples/type_checking/type_checking_example.py`
- [ ] Update this README with project-specific information
- [ ] Add examples to the CI workflow for automated validation
- [ ] Remove template placeholders and TODOs
- [ ] Document expected output clearly
- [ ] Link to your project's type checking documentation

## 🎯 Best Practices

1. **Keep Examples Simple**: Each example should demonstrate one concept clearly
2. **Use Real Types**: Show actual types from your project, not abstract examples
3. **Include Comments**: Explain why types are used, not just how
4. **Test Regularly**: Run MyPy on examples during development
5. **Show Both Success and Error Cases**: Demonstrate what works and what doesn't
6. **Document Type Errors**: Show common mistakes and how to fix them

## 🛠️ Customization Tips

### For Library Projects
- Show how consumers should use your public API with types
- Demonstrate Protocol implementations
- Include examples of generic type parameters

### For Application Projects  
- Show internal type usage patterns
- Demonstrate configuration type safety
- Include data validation examples

### For Framework Projects
- Show extension points with Protocol types
- Demonstrate plugin type interfaces
- Include middleware/handler typing examples

## 📚 Additional Resources

- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Typing Best Practices](https://typing.readthedocs.io/en/latest/)
