"""
input.py

Beginner's guide to the Python `input` function.

Overview:
---------
The `input()` function lets you get user input from the console. It pauses the program and waits for the user to type something and press Enter.

Basic Usage:
------------
input("prompt")

Parameters:
-----------
- prompt: (optional) A string displayed to the user before input.

Returns:
--------
- Always returns the user's input as a string.

"""
# Examples:
# ---------

# 1. Simple input
name = input("What is your name? ") # ask the user for the name
print("Hello,", name) # output Hello + the value that was put into the input buffer.

# 2. Input without prompt
data = input() # empty input prompt
print("You entered:", data) # output value 'data'

# 3. Convert input to integer
age = int(input("Enter your age: ")) # Converts the given input into an integer now giving a string or any other data type will result into an exception (error).
print("You will be", age + 1, "next year.") # output the given input age but + 1 more to arithmetic operations soon.

# 4. Convert input to float
height = float(input("Enter your height in meters: "))
print("Your height is", height, "meters.")

"""
Tips:
-----
- Use a prompt to tell the user what to enter.
- Always remember: input() returns a string. Convert to int or float if a number or decimal input is needed.
- Useful for interactive programs.

"""