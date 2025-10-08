"""
13-functions.py

Beginner's guide to Python functions.

Overview:
---------
A function is a block of code that performs a specific task. Functions help organize and reuse code.

Syntax:
-------
def function_name(parameters):
    # code block
    return value -> this is optional

Examples:
---------
"""

# 1. Simple function
def greet():
    print("Hello!")

greet()  # Output: Hello!
# When the function is called the code inside it will be executed, you can call a function as much as you want. greet() greet() greet().

# 2. Function with parameters
def add(a, b): # 2 parameters are taken a and b and when you call the function you pass their intended values. 
    return a + b

result = add(3, 5) # we declare a result variable and we assign the value of the variable to the add functions return value.
print(result)  # Output: 8

# 3. Function with default parameter
def say_hello(name="Guest"): # default value is guest if no value is given at function call.
    print(f"Hello, {name}!")

say_hello("Alice")  # Output: Hello, Alice!
say_hello()         # Output: Hello, Guest!

# 4. Function returning multiple values
def get_info():
    return "Alice", 30

name, age = get_info()
print(name, age)  # Output: Alice 30

"""
Tips:
-----
- Use functions to avoid repeating code.
- Use parameters to pass data into functions.
- Use return to send data back from functions.

"""