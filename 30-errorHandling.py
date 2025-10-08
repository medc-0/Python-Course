"""
30-errorHandling.py

Beginner's guide to advanced error handling in Python.

Overview:
---------
Error handling is crucial for creating robust applications. This guide covers advanced error handling techniques including custom exceptions, context managers, and best practices.

What is Error Handling?
-----------------------
Error handling is the process of anticipating, detecting, and responding to errors that occur during program execution. It helps create more reliable and user-friendly applications.

Key Concepts:
--------------
- Exception hierarchy: Understanding built-in exceptions
- Custom exceptions: Creating your own exception types
- Exception chaining: Preserving original exception information
- Context managers: Automatic resource cleanup
- Logging: Recording errors for debugging
- Best practices: Writing maintainable error handling

Examples:
---------
"""

import logging
import traceback
from contextlib import contextmanager
from typing import Optional

# 1. Basic exception handling
def basic_error_handling():
    try:
        result = 10 / 0
        print(f"Result: {result}")
    except ZeroDivisionError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("This always runs")

basic_error_handling()

# 2. Multiple exception handling
def multiple_exceptions():
    try:
        # This will raise a ValueError
        number = int("not_a_number")
    except ValueError as e:
        print(f"Value error: {e}")
    except TypeError as e:
        print(f"Type error: {e}")
    except Exception as e:
        print(f"Other error: {e}")
    else:
        print("No errors occurred")
    finally:
        print("Cleanup completed")

multiple_exceptions()

# 3. Custom exceptions
class CustomError(Exception):
    """Base class for custom exceptions"""
    pass

class ValidationError(CustomError):
    """Raised when validation fails"""
    def __init__(self, message, field=None):
        super().__init__(message)
        self.field = field

class DatabaseError(CustomError):
    """Raised when database operations fail"""
    def __init__(self, message, query=None):
        super().__init__(message)
        self.query = query

# Using custom exceptions
def validate_age(age):
    if not isinstance(age, int):
        raise ValidationError("Age must be an integer", field="age")
    if age < 0:
        raise ValidationError("Age cannot be negative", field="age")
    if age > 150:
        raise ValidationError("Age cannot exceed 150", field="age")
    return True

# Test custom exceptions
try:
    validate_age(-5)
except ValidationError as e:
    print(f"Validation error: {e} (Field: {e.field})")

# 4. Exception chaining
def function_a():
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        raise ValueError("Invalid calculation") from e

def function_b():
    try:
        function_a()
    except ValueError as e:
        raise RuntimeError("Operation failed") from e

# Test exception chaining
try:
    function_b()
except RuntimeError as e:
    print(f"Runtime error: {e}")
    print(f"Original error: {e.__cause__}")
    print(f"Chain: {e.__cause__.__cause__}")

# 5. Context managers for error handling
@contextmanager
def safe_file_operation(filename, mode):
    file = None
    try:
        file = open(filename, mode)
        yield file
    except FileNotFoundError:
        print(f"File {filename} not found")
        yield None
    except PermissionError:
        print(f"Permission denied for {filename}")
        yield None
    finally:
        if file:
            file.close()

# Using context manager
with safe_file_operation("nonexistent.txt", "r") as file:
    if file:
        content = file.read()
        print(f"Content: {content}")

# 6. Logging errors
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('errors.log'),
        logging.StreamHandler()
    ]
)

def risky_operation():
    try:
        result = 10 / 0
        return result
    except ZeroDivisionError as e:
        logging.error(f"Division by zero: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

# Test logging
try:
    risky_operation()
except ZeroDivisionError:
    print("Error logged to file")

# 7. Retry mechanism
import time
import random

def retry_operation(max_retries=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_operation(max_retries=3, delay=0.5)
def unreliable_operation():
    if random.random() < 0.7:  # 70% chance of failure
        raise Exception("Random failure")
    return "Success!"

# Test retry mechanism
try:
    result = unreliable_operation()
    print(f"Operation result: {result}")
except Exception as e:
    print(f"Operation failed after retries: {e}")

# 8. Exception handling in classes
class Calculator:
    def __init__(self):
        self.history = []
    
    def divide(self, a, b):
        try:
            result = a / b
            self.history.append(f"{a} / {b} = {result}")
            return result
        except ZeroDivisionError:
            self.history.append(f"{a} / {b} = Error: Division by zero")
            raise
        except TypeError:
            self.history.append(f"{a} / {b} = Error: Invalid types")
            raise
    
    def get_history(self):
        return self.history

# Using calculator with error handling
calc = Calculator()
try:
    result = calc.divide(10, 2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

try:
    result = calc.divide(10, 0)
except Exception as e:
    print(f"Error: {e}")

print(f"History: {calc.get_history()}")

# 9. Exception handling in generators
def safe_generator(data):
    for item in data:
        try:
            yield item * 2
        except Exception as e:
            print(f"Error processing {item}: {e}")
            continue

# Using safe generator
data = [1, 2, "three", 4, 5]
for result in safe_generator(data):
    print(f"Result: {result}")

# 10. Exception handling in async code
import asyncio

async def async_operation():
    try:
        # Simulate async operation
        await asyncio.sleep(0.1)
        result = 10 / 0
        return result
    except ZeroDivisionError as e:
        print(f"Async error: {e}")
        raise

# Run async operation
async def main():
    try:
        result = await async_operation()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Async error caught: {e}")

# asyncio.run(main())

# 11. Exception handling with context managers
class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connected = False
    
    def __enter__(self):
        try:
            # Simulate connection
            if self.host == "invalid":
                raise ConnectionError("Invalid host")
            self.connected = True
            print(f"Connected to {self.host}:{self.port}")
            return self
        except Exception as e:
            print(f"Connection failed: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connected:
            print(f"Disconnected from {self.host}:{self.port}")
        if exc_type:
            print(f"Exception in context: {exc_val}")
        return False  # Don't suppress exceptions
    
    def query(self, sql):
        if not self.connected:
            raise RuntimeError("Not connected to database")
        return f"Executing: {sql}"

# Using database connection
try:
    with DatabaseConnection("localhost", 5432) as db:
        result = db.query("SELECT * FROM users")
        print(result)
except Exception as e:
    print(f"Database error: {e}")

# 12. Exception handling with decorators
def handle_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return None
    return wrapper

@handle_exceptions
def risky_function(x, y):
    return x / y

# Test decorated function
result = risky_function(10, 0)
print(f"Result: {result}")

# 13. Exception handling in file operations
def safe_file_read(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except PermissionError:
        print(f"Permission denied for {filename}")
        return None
    except Exception as e:
        print(f"Unexpected error reading {filename}: {e}")
        return None

# Test file reading
content = safe_file_read("nonexistent.txt")
if content:
    print(f"Content: {content}")

# 14. Exception handling in data validation
class DataValidator:
    @staticmethod
    def validate_email(email):
        if not isinstance(email, str):
            raise ValidationError("Email must be a string")
        if "@" not in email:
            raise ValidationError("Email must contain @")
        if "." not in email.split("@")[1]:
            raise ValidationError("Email must contain domain")
        return True
    
    @staticmethod
    def validate_phone(phone):
        if not isinstance(phone, str):
            raise ValidationError("Phone must be a string")
        if not phone.replace("-", "").replace(" ", "").isdigit():
            raise ValidationError("Phone must contain only digits")
        return True

# Test data validation
def validate_user_data(email, phone):
    errors = []
    
    try:
        DataValidator.validate_email(email)
    except ValidationError as e:
        errors.append(f"Email error: {e}")
    
    try:
        DataValidator.validate_phone(phone)
    except ValidationError as e:
        errors.append(f"Phone error: {e}")
    
    if errors:
        raise ValidationError("; ".join(errors))
    
    return True

# Test validation
try:
    validate_user_data("invalid-email", "123-456-7890")
except ValidationError as e:
    print(f"Validation errors: {e}")

# 15. Exception handling best practices
class ErrorHandler:
    @staticmethod
    def handle_exception(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError as e:
                logging.error(f"Validation error: {e}")
                return {"error": "Invalid data", "details": str(e)}
            except DatabaseError as e:
                logging.error(f"Database error: {e}")
                return {"error": "Database operation failed", "details": str(e)}
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                return {"error": "Internal server error", "details": "An unexpected error occurred"}
        return wrapper

@ErrorHandler.handle_exception
def process_user_data(user_data):
    # Simulate processing
    if not user_data.get("name"):
        raise ValidationError("Name is required")
    if not user_data.get("email"):
        raise ValidationError("Email is required")
    
    # Simulate database operation
    if user_data.get("email") == "error@example.com":
        raise DatabaseError("Database connection failed")
    
    return {"status": "success", "user_id": 123}

# Test error handling
test_cases = [
    {"name": "John", "email": "john@example.com"},
    {"name": "Jane", "email": "error@example.com"},
    {"name": "", "email": "jane@example.com"},
    {}
]

for user_data in test_cases:
    result = process_user_data(user_data)
    print(f"Result for {user_data}: {result}")

"""
Error Handling Best Practices:
------------------------------
1. Use specific exception types
2. Don't catch all exceptions unless necessary
3. Log errors with context
4. Use finally blocks for cleanup
5. Don't suppress exceptions silently
6. Use custom exceptions for business logic
7. Handle exceptions at appropriate levels
8. Use context managers for resource management
9. Test error handling code
10. Document expected exceptions

Common Exception Types:
----------------------
- ValueError: Invalid value
- TypeError: Wrong type
- KeyError: Missing key
- IndexError: Invalid index
- FileNotFoundError: File not found
- PermissionError: Access denied
- ConnectionError: Network issues
- RuntimeError: General runtime errors

When to Use Error Handling:
--------------------------
- User input validation
- File operations
- Network operations
- Database operations
- External API calls
- Resource management

When NOT to Use Error Handling:
-------------------------------
- For control flow
- To hide bugs
- For performance optimization
- When exceptions are expected
- For input validation (use validation libraries)

Tips:
-----
- Always log errors with context
- Use specific exception types
- Don't catch and ignore exceptions
- Use finally blocks for cleanup
- Test error scenarios
- Document expected exceptions
- Use context managers for resources
- Consider using exception chaining

"""
