"""
17-exceptionHandling.py

Beginner's guide to exception handling in Python.

Overview:
---------
Exception handling lets you deal with errors gracefully using try, except, finally.

Examples:
---------
"""

# Basic try-except
try:
    x = 1 / 0
except ZeroDivisionError:
    print("You can't divide by zero!")  # Output: You can't divide by zero!

# Catching any error
try:
    res = 1+1
   # print(unknown_var + res) -> will result in an Exception because that variable doesnt even exist.
except Exception as e:
    print("Error:", e)

# Using finally
try:
    print("Try block")
finally:
    print("This always runs.")

"""
Tips:
-----
- Use try-except to prevent your program from crashing.
- Catch specific exceptions for better error handling.
- finally block always runs, even if there was an error.
- Always Try to use Try-Except handling when you work with user-input.

"""