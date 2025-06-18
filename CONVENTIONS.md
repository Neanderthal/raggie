# Python Code Conventions Guide

## 1. Formatting & Layout

### 1.1 Indentation
- Use **4 spaces** per indentation level.
- **Tabs are not allowed**.

### 1.2 Line Length
- Recommended: **≤ 88 characters**.
- Acceptable range: **88–120 characters** with justification.
- Hard limit: **120** characters.

### 1.3 Blank Lines
- **2 blank lines** between top-level functions or class definitions.
- **1 blank line** between logically related code sections within functions.

### 1.4 Whitespace Rules
- No whitespace inside parentheses, brackets, or braces:
```python
  call(a, b)
  ```
  ```
```

* Use a single space:

  * Around binary operators (`=`, `+`, `-`, etc.).
  * After commas, colons, and semicolons.
  * After `#` in inline comments.

---

## 2. Imports

### 2.1 Grouping (in order, separated by blank lines)

1. **Standard library**
2. **Third-party packages**
3. **Local application or library code**

### 2.2 Import Style

* Use **absolute imports** only.
* Do **not** use `from module import *`.
* Sort imports using `isort` or `ruff`.

---

## 3. Naming Conventions

| Type            | Convention                          | Example                 |
| --------------- | ----------------------------------- | ----------------------- |
| Variables       | `snake_case`                        | `user_id`, `config_map` |
| Functions       | `snake_case`                        | `load_data()`           |
| Classes         | `PascalCase`                        | `DataLoader`, `User`    |
| Constants       | `UPPER_SNAKE_CASE`                  | `DEFAULT_TIMEOUT`       |
| Private Members | `_single_leading_underscore`        |                         |
| Internal Use    | `__double_leading_underscore__`     |                         |
| Avoid           | Single-letter names (`l`, `O`, `I`) |                         |

---

## 4. Docstrings & Comments

### 4.1 Docstrings

* Required for **all public** classes, functions, and modules.
* Follow [PEP 257](https://peps.python.org/pep-0257/).
* Use triple double-quotes (`""" """`).

### 4.2 Comments

* Comments should be **clear**, **concise**, and **grammatically correct**.
* Use **block comments** (`# `) aligned with code.
* **Inline comments** should be used sparingly.

---

## 5. Pythonic Idioms

### 5.1 Use Built-ins

Prefer built-in functions and idioms:

```python
# Good
sum(values)
any(x > 10 for x in items)
[x for x in items if x > 10]
```

### 5.2 Prefer Guard Clauses

```python
# Good
def process(item):
    if not item:
        return None
    return compute(item)
```

### 5.3 Avoid Deep Nesting

* Keep functions **flat and readable**.
* Break out logic into helper functions.

---

## 6. Typing & Type Checking

### 6.1 Type Hints

* Use [PEP 484](https://peps.python.org/pep-0484/) style annotations for **all public functions**, including return types.

```python
def send_email(to: str, subject: str, body: str) -> bool:
```

### 6.2 Type Checkers

* Run `mypy` or `pyright` in CI or locally for type enforcement.

---

## 7. Tools & Automation

### 7.1 Formatters & Linters

* **Formatter**: `black` (autoformat on save or commit)
* **Linter**: `ruff`, `pyright`
* **Import Sorter**: `isort` (or handled by `ruff`)
* **Unused Remover**: `autoflake`

### 7.2 Pre-commit Hooks

Use [`pre-commit`](https://pre-commit.com/) to enforce linting, formatting, and type checks.

---

## 8. Testing

* All new logic must be covered by **unit tests**.
* Use `pytest` as the default test framework.
* Include edge cases, exception handling, and meaningful test names.
* Aim for high **coverage** but prioritize **quality of test logic** over coverage percentage.

---

## 9. Project Structure

* Organize by **domain or feature**, not layer.
* One class/function per module when possible.
* Keep `__main__` logic minimal:

  ```python
  if __name__ == "__main__":
      main()
  ```
* Use `logging` instead of `print()`.

---

## 10. LLM-Specific Instructions

* Always adhere to the above conventions when generating or editing Python code.
* When refactoring:

  * Detect and eliminate code smells (duplication, long functions, deep nesting).
  * Prefer extract method/class, polymorphism over conditionals, and encapsulated data structures.
* Use Black-compatible formatting.
* Prefer standard library and built-in idioms unless specified otherwise.

--
