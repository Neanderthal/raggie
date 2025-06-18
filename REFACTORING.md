A comprehensive instruction set for LLMs to perform safe, structured, idiomatic refactoring of Python codebases.

---

## ✅ Goals of Refactoring

* **Improve Maintainability**: Simplify code structure and naming.
* **Preserve Behavior**: Do not change what the code does.
* **Enhance Readability & Testability**: Clear separation of concerns.
* **Leverage Pythonic Idioms**: Replace imperative code with concise built-ins and idioms.

---

## 🧪 Prerequisite: Test Coverage Required

> **ALWAYS verify test coverage before refactoring.** Refactor only when tests exist or after adding sufficient tests.

### Verify:

* Unit test coverage of functions/classes
* Integration test coverage for workflows
* Use of frameworks: `pytest`, `unittest`, `nose2`

---

## 👃 Common Code Smells to Eliminate

| Smell                        | Fix Pattern                                 |
| ---------------------------- | ------------------------------------------- |
| Long Methods / Large Classes | Extract Method / Class                      |
| Duplicated Code              | Extract Function / DRY                      |
| Magic Numbers                | Replace with Named Constants                |
| Deep Nesting                 | Use Guard Clauses / Simplify Logic          |
| Primitive Obsession          | Introduce Parameter or Data Object          |
| Data Clumps                  | Use a structured object (e.g. `@dataclass`) |
| Conditional Complexity       | Replace with Polymorphism                   |
| Shotgun Surgery              | Consolidate Responsibility                  |

---

## 🔁 Refactoring Patterns & Examples

### 1. Extract Method / Class

**When**: Function does too much.

```python
# BEFORE
def process_order(data):
    validate(data)
    shipping = calculate_shipping(data)
    save_to_db(data, shipping)
    notify_user(data)

# AFTER
def process_order(data):
    validate(data)
    shipping = _calculate_shipping(data)
    _persist_order(data, shipping)
    _send_confirmation(data)
```

---

### 2. Rename for Clarity

**When**: Poorly named variables or functions.

```python
# BEFORE
MAX = 365
def calc(x): return x * MAX

# AFTER
DAYS_IN_YEAR = 365
def calculate_age_in_days(age_years): return age_years * DAYS_IN_YEAR
```

---

### 3. Replace Conditionals with Polymorphism

**When**: Large `if-elif` chains for types or states.

```python
# BEFORE
def area(shape):
    if shape.type == 'circle':
        return math.pi * shape.radius**2

# AFTER
class Shape: def area(self): raise NotImplementedError
class Circle(Shape): def area(self): return math.pi * self.radius**2
```

---

### 4. Introduce Parameter Object

**When**: Many function parameters or related arguments.

```python
# BEFORE
def send_email(to, subject, body, retries, timeout): ...

# AFTER
@dataclass
class EmailConfig:
    retries: int
    timeout: float

def send_email(to, subject, body, config: EmailConfig): ...
```

---

### 5. Encapsulate Collection

**When**: Internal collections are mutated externally.

```python
# Wrap mutation in methods that enforce invariants.
```

---

## 🐍 Python-Specific Transformations

Use the following idioms to improve clarity and reduce boilerplate:

| Old Style                     | Pythonic Replacement                        |
| ----------------------------- | ------------------------------------------- |
| Manual loops                  | List/Dict/Set comprehensions                |
| Accumulation                  | `sum()`, `min()`, `max()`, `any()`, `all()` |
| `for i in range(len(...))`    | `enumerate()`                               |
| Multiple iterables            | `zip()`                                     |
| Repetitive logic              | Use `@property`, `@lru_cache`, etc.         |
| Try-finally resource handling | `with` context manager                      |
| Assignment in conditions      | `:=` (Python 3.8+)                          |

---

## 🔁 Refactoring Workflow for LLMs

1. **Detect Smell**: Identify code issues based on above list.
2. **Verify Tests Exist**: Confirm coverage or halt and suggest writing tests.
3. **Choose Appropriate Pattern**: Based on the smell.
4. **Apply Refactor in Steps**: Maintain behavioral parity.
5. **Verify Output Diff**: Ensure no unintended changes.
6. **Generate Clean Commit Message**:

   > `Extract validate_input() from validate_all()`
7. **Repeat Until Codebase Clean**

---

## 🔧 Tooling & Automation (LLMs: Recommend as Needed)

* Linters: `flake8`, `ruff`
* Formatters: `black`, `isort`
* Type Checkers: `mypy`
* Refactoring Engines: Rope, Sourcery
* Editor Plugins: pylsp-rope, VS Code Python Extension

---

## 🧠 Worked Examples

### Flatten Nested Logic

```python
# BEFORE
def process(items):
    results = []
    for item in items:
        if item.active:
            if item.value > 0:
                results.append(transform(item))

# AFTER
def process(items):
    return [transform(i) for i in items if i.active and i.value > 0]
```

---

### Strategy Pattern for Duplication

```python
# BEFORE
def calculate_discount(order, customer):
    if customer.type == 'vip': return order.total * 0.2

# AFTER
class DiscountStrategy: def discount(self, order): ...
class VIPDiscount(DiscountStrategy): def discount(self, order): return order.total * 0.2
```

---

## 🧩 Final Reminders

* **Never refactor without tests**
* **Prefer readability over cleverness**
* **Apply changes incrementally**
* **Document with clear diffs and messages**
* **Use Python idioms whenever possible**


