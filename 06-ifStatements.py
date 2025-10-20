"""
06-ifStatements.py

Beginner's guide to Python if, elif, and else statements.

Overview:
---------
Conditional statements let you run code *ONLY* if certain conditions are true.

Syntax:
-------
if condition:
    # code runs if condition is True
elif another_condition:
    # code runs if previous conditions are False and this is True
else:
    # code runs if none of the above conditions are True
    
Examples:
---------
"""

# 1. Simple if statement
age = 18
if age >= 18:
    print("You are an adult.")  # Output: You are an adult.

# 2. if-else statement
number = 5
if number % 2 == 0:
    print("Even number")
else:
    print("Odd number")  # Output: Odd number

# 3. if-elif-else statement
score = 85
if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")  # Output: Grade: B
else:
    print("Grade: C")

# 4. Multiple elifs
temperature = 30
if temperature > 35:
    print("It's very hot!")
elif temperature > 25:
    print("It's warm.")  # Output: It's warm.
elif temperature > 15:
    print("It's cool.")
else:
    print("It's cold.")

"""
Tips:
-----
- Use indentation (spaces) to define code blocks.
- You can have multiple elif statements.
- else is optional.

"""