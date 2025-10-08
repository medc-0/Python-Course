"""
print.py

Beginner's guide to the Python `print` function.

Overview:
---------
The `print()` function displays output to the console. It's used to show text, numbers, and other objects and its mostly used for debugging purposes and to output text or values.

Basic Usage:
------------
print(*objects, sep=' ', end='\n')

Parameters:
-----------
- *objects: Items to print.
- sep: Separator between items (default: space).
- end: What to print at the end (default: newline).

F-Strings:
----------
F-strings (formatted string literals) are a way to embed expressions inside string constants, using curly braces `{}`.
Prefix the string with `f` or `F` and put variables or expressions inside `{}` to display their values.

Example:
name = "Alice"
age = 30
print(f"Name: {name}, Age: {age}")  # Output: Name: Alice, Age: 30
"""

# Examples:
# ---------

# 1. Print a string
print("Hello, world!")  # Output: Hello, world!

# 2. Print multiple items
print("Hello", "world", 123)  # Output: Hello world 123

# 3. Custom separator
print("apple", "banana", "cherry", sep=", ")  # Output: apple, banana, cherry

# 4. Custom end
print("Hello", end="!")  # Output: Hello!
print("Next")            # Output: Next

# 5. Empty Print statement (for space)
print()

# 6. F-string example
name = "Alice" # This is a variable more to this in the next lesson.
age = 30 # another variable more to this next lesson.
print(f"Name: {name}, Age: {age}")  # Output: Name: Alice, Age: 30

"""
Tips:
-----
- Use `sep` and `end` for formatting.
- Use `empty` print statements for space.
- Use f-strings for readable and dynamic output.

"""