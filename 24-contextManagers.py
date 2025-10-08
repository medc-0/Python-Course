"""
24-contextManagers.py

Beginner's guide to Python context managers.

Overview:
---------
Context managers are objects that define what happens when you enter and exit a context (like opening and closing a file). They use the `with` statement and ensure proper resource management.

What are Context Managers?
--------------------------
Context managers are objects that implement the context manager protocol, which consists of `__enter__` and `__exit__` methods. They ensure that resources are properly acquired and released.

Key Concepts:
--------------
- `with` statement: Used to enter and exit a context
- `__enter__`: Called when entering the context
- `__exit__`: Called when exiting the context
- Automatic resource cleanup
- Exception handling in context

Examples:
---------
"""

# 1. Basic file context manager (built-in)
# This is the most common use of context managers
with open("example.txt", "w") as file:
    file.write("Hello, World!")
# File is automatically closed here

# 2. Custom context manager using class
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.filename)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing file: {self.filename}")
        if self.file:
            self.file.close()
        # Return False to propagate exceptions, True to suppress them
        return False

# Using the custom context manager
try:
    with FileManager("test.txt", "w") as file:
        file.write("This is a test file")
        print("File written successfully")
except Exception as e:
    print(f"Error: {e}")

# 3. Context manager using contextlib.contextmanager
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    print(f"Opening file: {filename}")
    file = open(filename, mode)
    try:
        yield file
    finally:
        print(f"Closing file: {filename}")
        file.close()

# Using the context manager
with file_manager("example.txt", "w") as file:
    file.write("Hello from context manager!")

# 4. Context manager for timing operations
import time

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        print(f"Starting timer: {self.name}")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"Timer {self.name} completed in {duration:.4f} seconds")

# Using the timer context manager
with Timer("slow operation"):
    time.sleep(1)  # Simulate slow operation
    print("Operation completed")

# 5. Context manager for database connections
class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to database at {self.host}:{self.port}")
        # Simulate database connection
        self.connection = f"Connection to {self.host}:{self.port}"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing database connection")
        self.connection = None

# Using the database context manager
with DatabaseConnection("localhost", 5432) as db:
    print(f"Using database: {db}")

# 6. Context manager for temporary directory
import os
import tempfile
import shutil

class TemporaryDirectory:
    def __init__(self, prefix="temp_"):
        self.prefix = prefix
        self.temp_dir = None
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix=self.prefix)
        print(f"Created temporary directory: {self.temp_dir}")
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Removed temporary directory: {self.temp_dir}")

# Using the temporary directory context manager
with TemporaryDirectory("my_temp_") as temp_dir:
    # Create a file in the temporary directory
    file_path = os.path.join(temp_dir, "test.txt")
    with open(file_path, "w") as f:
        f.write("Temporary file content")
    print(f"Created file: {file_path}")

# 7. Context manager for changing directory
class ChangeDirectory:
    def __init__(self, new_dir):
        self.new_dir = new_dir
        self.original_dir = None
    
    def __enter__(self):
        self.original_dir = os.getcwd()
        print(f"Changing directory from {self.original_dir} to {self.new_dir}")
        os.chdir(self.new_dir)
        return self.new_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Changing directory back to {self.original_dir}")
        os.chdir(self.original_dir)

# Using the change directory context manager
original_dir = os.getcwd()
print(f"Original directory: {original_dir}")

# This would work if the directory exists
# with ChangeDirectory("/tmp"):
#     print(f"Current directory: {os.getcwd()}")

# 8. Context manager for resource locking
import threading

class LockManager:
    def __init__(self, lock):
        self.lock = lock
    
    def __enter__(self):
        print("Acquiring lock")
        self.lock.acquire()
        return self.lock
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Releasing lock")
        self.lock.release()

# Using the lock context manager
lock = threading.Lock()
with LockManager(lock):
    print("Critical section - lock is held")
    # Do some work here

# 9. Context manager with exception handling
class SafeOperation:
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.success = False
    
    def __enter__(self):
        print(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print(f"{self.operation_name} completed successfully")
            self.success = True
        else:
            print(f"{self.operation_name} failed with error: {exc_val}")
            self.success = False
        # Don't suppress exceptions
        return False

# Using the safe operation context manager
with SafeOperation("risky operation") as op:
    # This will succeed
    result = 10 / 2
    print(f"Result: {result}")

# This will fail
try:
    with SafeOperation("another risky operation") as op:
        result = 10 / 0  # This will raise an exception
except ZeroDivisionError:
    print("Caught division by zero error")

# 10. Multiple context managers
class Resource1:
    def __enter__(self):
        print("Acquiring resource 1")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource 1")

class Resource2:
    def __enter__(self):
        print("Acquiring resource 2")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource 2")

# Using multiple context managers
with Resource1() as r1, Resource2() as r2:
    print("Using both resources")

"""
Built-in Context Managers:
--------------------------
- open(): File operations
- threading.Lock(): Thread synchronization
- decimal.localcontext(): Decimal precision
- warnings.catch_warnings(): Warning handling
- contextlib.suppress(): Exception suppression
- contextlib.redirect_stdout(): Output redirection

Context Manager Benefits:
------------------------
- Automatic resource cleanup
- Exception safety
- Cleaner code
- Reduced boilerplate
- Better error handling

When to Use Context Managers:
-----------------------------
- File operations
- Database connections
- Network connections
- Locking mechanisms
- Temporary resources
- Configuration changes

When NOT to Use Context Managers:
---------------------------------
- Simple operations that don't need cleanup
- When you need fine-grained control
- When the context is very short-lived
- When you need to access resources outside the context

Tips:
-----
- Always use `with` statements for file operations
- Implement both `__enter__` and `__exit__` methods
- Handle exceptions properly in `__exit__`
- Use `contextlib.contextmanager` for simple cases
- Consider using `contextlib.suppress` for exception handling
- Test context managers with exceptions

"""
