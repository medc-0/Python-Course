"""
variables.py

Beginner's guide to Python data types and variables.

Data Types:
-----------
A data type defines the kind of value a variable can hold and how it behaves.

Common Python data types:
- int: Integer numbers (e.g., 5, -3, 42)
- float: Decimal numbers (e.g., 3.14, -0.5)
- str: Strings, sequences of characters (e.g., "hello", 'world')
- bool: Boolean values (True or False)

type() Function:
----------------
The built-in `type()` function shows the data type of a value or variable.

Example:
x = 10
print(type(x))  # Output: <class 'int'> this means its a 'number which is in python a integer.'

Variables:
----------
A variable is a name that refers to a value. You can store any data type in a variable.

Declaring and Assigning Variables:
----------------------------------
Step 1: Declare a variable name
Step 2: Assign a value using =

Example:
# Step 1: Declare variable name
age
# Step 2: Assign value
age = 25

# You can do both steps in one line (as is usual in Python):
age = 25

More Examples:
name = "Alice"
height = 1.75
is_student = True

"""

# Examples:
# ---------

# Assigning values to variables
age = 25
height = 1.75
greeting = "Hello!"
is_student = False

# Using variables
print(age)         # Output: 25
print(height)      # Output: 1.75
print(greeting)    # Output: Hello!
print(is_student)  # Output: False

# Variables can change value
age = 26 # change the value of the declared age variable
print(age) # Output: 26

# Using type() to check variable types
print(type(age))       # Output: <class 'int'>
print(type(height))    # Output: <class 'float'>
print(type(greeting))  # Output: <class 'str'>
print(type(is_student))# Output: <class 'bool'>

"""
Tips:
-----
- Variable names should be descriptive.
- Python is dynamically typed: you don't need to declare the type.
- You can assign any type to any variable.
- Use type() to check the type of a variable.

"""