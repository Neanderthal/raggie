Here is **The Ultimate Rulebook for Handling Exceptions in Python**, distilled from the article you provided and enhanced with best practices from the Python community:
[[The Ultimate Guide to Error Handling in Python (article)]]
---

## 🔰 FOUNDATIONAL PRINCIPLES

### 1. **Prefer EAFP over LBYL**

- **EAFP (Easier to Ask Forgiveness than Permission)**: Do the operation and handle exceptions if they arise.
    
- **LBYL (Look Before You Leap)**: Check all possible conditions before acting — but this is more error-prone and can introduce race conditions.
    

> ✅ _Use EAFP when interacting with external systems or resources._

---

## 🔍 CATEGORIZE EVERY EXCEPTION

### 2. **Classify Exceptions by Origin**

- **New Error**: Created by your own code due to internal validation or business logic.
    
- **Bubbled-up Error**: Raised from a lower-level function that your code called.
    

### 3. **Classify Exceptions by Recoverability**

- **Recoverable**: You know how to fix it and can continue.
    
- **Non-Recoverable**: You can't fix it — propagate it up.
    

---

## 🔄 4 TYPES OF EXCEPTION HANDLING (The Matrix)

|Type|Action to Take|
|---|---|
|🆕 New + ✅ Recoverable|Fix internally, continue|
|🔁 Bubbled-up + ✅ Recoverable|Catch, fix, continue|
|🆕 New + ❌ Non-Recoverable|Raise a new exception|
|🔁 Bubbled-up + ❌ Non-Recoverable|**Let it bubble up**|

---

## ⚙️ TACTICAL GUIDELINES

### 4. **Never Catch What You Can’t Handle**

- Avoid `except Exception` or `except:` unless you’re at the **top-level boundary** of your app.
    
- Catch **specific exception classes** that you're explicitly prepared to deal with.
    

### 5. **Use Custom Exceptions When Appropriate**

- Create meaningful exception classes like `ValidationError`, `DatabaseUnavailableError`, etc.
    

```python
class ValidationError(Exception):
    pass
```

---

## 💡 BEST PRACTICES BY SCENARIO

### 6. **Top-Level Exception Catching**

- Only here is catching all exceptions acceptable — for logging, cleanup, and graceful exit.
    

```python
try:
    main()
except Exception as e:
    log(e)
    sys.exit(1)
```

### 7. **Use Frameworks’ Exception Handling**

- **Flask, Django, FastAPI** etc. provide global handlers. Let them handle low-level exceptions.
    
- Avoid per-view/manual exception catching unless recovery is meaningful.
    

### 8. **Design for Separation of Concerns**

- Avoid mixing error handling and UI/UX concerns.
    
- Let exceptions bubble to a higher layer that knows how to present the issue.
    

---

## 🧪 ENVIRONMENT-SPECIFIC HANDLING

### 9. **Different Behaviors for Dev and Prod**

```python
if mode == "development":
    raise
else:
    log(e)
    sys.exit(1)
```

> ❗️Crashes during development are helpful — don’t hide them!

---

## 📋 CODE CLEANLINESS AND LOGGING

### 10. **Use `logger.exception()` not `logger.error()`**

- `logger.exception()` auto-includes the stack trace.
    

### 11. **Avoid Redundant Error Handling**

- Don’t catch and log an error just to re-raise it.
    

### 12. **Let Frameworks Handle Common Patterns**

- E.g., Flask-SQLAlchemy handles rollbacks automatically after exceptions — no need for manual rollback logic in every route.
    

---

## 🔒 RARELY OKAY PRACTICES

### 13. **When Catching All Exceptions Is Acceptable**

- **CLI app main()**
    
- **Event loop/tkinter/GUI handlers**
    
- **Background jobs or async workers**
    

---

## 🧠 MENTAL MODELS

### 14. **Think in Terms of State Correction**

- If the state can be corrected, don’t raise — fix and move on.
    
- If it cannot, raise immediately with relevant context.
    

### 15. **Exceptions Are Contracts**

- Make functions raise predictable, documented exceptions.
    
- Don't use exceptions for control flow unless unavoidable.
    

---

## ✅ CHECKLIST BEFORE RAISING OR CATCHING

-  Is this a new error or a bubbled-up one?
    
-  Can I recover from it here?
    
-  Am I catching the most specific exception class?
    
-  Will this error need to be logged?
    
-  Do I understand the framework’s error lifecycle?
    
-  Am I in development or production mode?
    

---

Would you like this turned into a printable markdown or PDF cheat sheet?For more detailed information, see:
[[The Ultimate Guide to Error Handling in Python (article)]]
