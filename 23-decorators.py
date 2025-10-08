"""
23-decorators.py

Beginner's guide to Python decorators.

Overview:
---------
Decorators are a powerful feature in Python that allow you to modify or enhance functions without changing their code. They use the `@` symbol and are commonly used for logging, timing, authentication, and more.

What are Decorators?
--------------------
Decorators are functions that take another function as input and return a modified version of that function. They provide a clean way to add functionality to existing functions.

Key Concepts:
--------------
- Decorators are functions that modify other functions
- Use the `@` symbol to apply decorators
- Decorators can accept arguments
- Multiple decorators can be applied to one function
- Decorators preserve function metadata

Examples:
---------
"""

# 1. Basic decorator
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# Using the decorated function
say_hello()

# 2. Decorator with function arguments
def decorator_with_args(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper

@decorator_with_args
def add_numbers(a, b):
    return a + b

# Using the decorated function
result = add_numbers(5, 3)
print(f"Result: {result}")

# 3. Decorator that measures execution time
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)  # Simulate slow operation
    return "Done!"

# Using the decorated function
result = slow_function()

# 4. Decorator with parameters
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(times):
                print(f"Call {i+1}:")
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

# Using the decorated function
greet("Alice")

# 5. Decorator for logging
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"LOG: Calling {func.__name__} with arguments {args}")
        try:
            result = func(*args, **kwargs)
            print(f"LOG: {func.__name__} completed successfully")
            return result
        except Exception as e:
            print(f"LOG: {func.__name__} failed with error: {e}")
            raise
    return wrapper

@log_calls
def divide_numbers(a, b):
    return a / b

# Using the decorated function
try:
    result = divide_numbers(10, 2)
    print(f"Division result: {result}")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# 6. Decorator for authentication
def require_auth(func):
    def wrapper(*args, **kwargs):
        # Simulate authentication check
        is_authenticated = True  # In real app, this would check actual auth
        if is_authenticated:
            print("Authentication successful")
            return func(*args, **kwargs)
        else:
            print("Authentication failed")
            return None
    return wrapper

@require_auth
def sensitive_operation():
    return "Sensitive data accessed"

# Using the decorated function
result = sensitive_operation()

# 7. Multiple decorators
def bold(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return f"<b>{result}</b>"
    return wrapper

def italic(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return f"<i>{result}</i>"
    return wrapper

@bold
@italic
def get_text():
    return "Hello, World!"

# Using the decorated function
formatted_text = get_text()
print(f"Formatted text: {formatted_text}")

# 8. Class-based decorator
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Function {self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def example_function():
    return "Function executed"

# Using the decorated function
example_function()
example_function()
example_function()

# 9. Decorator that preserves function metadata
from functools import wraps

def preserve_metadata(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@preserve_metadata
def example_func():
    """This is an example function."""
    return "Example"

# Check if metadata is preserved
print(f"Function name: {example_func.__name__}")
print(f"Function docstring: {example_func.__doc__}")

# 10. Decorator for caching (memoization)
def memoize(func):
    cache = {}
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            print(f"Computing {func.__name__} for {args}")
            cache[key] = func(*args, **kwargs)
        else:
            print(f"Using cached result for {func.__name__} with {args}")
        return cache[key]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Using the decorated function
print(f"Fibonacci(10): {fibonacci(10)}")
print(f"Fibonacci(10) again: {fibonacci(10)}")  # Uses cache

"""
Common Decorator Patterns:
-------------------------
1. Logging: Track function calls and results
2. Timing: Measure execution time
3. Authentication: Check user permissions
4. Caching: Store results to avoid recomputation
5. Validation: Check input parameters
6. Retry: Retry failed operations
7. Rate limiting: Control function call frequency

Built-in Decorators:
--------------------
- @property: Create getter methods
- @staticmethod: Create static methods
- @classmethod: Create class methods
- @functools.wraps: Preserve function metadata
- @functools.lru_cache: Caching decorator

When to Use Decorators:
-----------------------
- Adding cross-cutting concerns (logging, timing)
- Implementing authentication/authorization
- Caching expensive computations
- Input validation
- Retry mechanisms
- Rate limiting

When NOT to Use Decorators:
---------------------------
- When the functionality is specific to one function
- When it makes code harder to understand
- When you need to modify function behavior significantly
- When performance is critical (decorators add overhead)

Tips:
-----
- Use @functools.wraps to preserve function metadata
- Keep decorators simple and focused
- Document decorator behavior
- Test decorators thoroughly
- Consider using functools.partial for parameterized decorators

"""
