"""
15-imports.py

Beginner's guide to importing modules in Python.

Overview:
---------
Imports let you use code from other files or libraries.

Examples:
---------
"""

# Import a built-in module
import math
print(math.sqrt(16))  # Output: 4.0

# Import specific functions
from math import pi
print(pi)  # Output: 3.141592653589793

# Import with alias
import datetime as dt
print(dt.datetime.now())

# Import your own file (if you have mymodule.py)
# import mymodule

# Importing the math module
import math

# Using a function from the math module
result = math.sqrt(25)
print(result)  # Output: 5.0

# Importing specific functions from a module
from math import pow, ceil

# Using the imported functions
print(pow(2, 3))  # Output: 8.0
print(ceil(4.2))  # Output: 5

# Importing a module with an alias
import statistics as stats

# Using a function from the aliased module
data = [1, 2, 3, 4, 5]
print(stats.mean(data))  # Output: 3

"""
Tips:
-----
- Organize your code into modules for better structure.
- Use built-in modules to save time and effort.
- Remember to install any external modules you want to use.
- Use imports to organize code and reuse functions.
- Built-in modules provide extra features (math, random, datetime, etc.).
- Use `as` to give a module a shorter name.

"""