"""
08-forLoops.py

Beginner's guide to Python for loops.

Overview:
---------
A for loop repeats code for each item in a sequence (like a list, string, or range). It's useful when you know how many times you want to repeat.

Syntax:
-------
for variable in sequence:
    # code to run for each item

Examples:
---------
"""

# 1. Loop through a list
fruits = ["apple", "banana", "cherry"] # this is a list will be explained in the next lesson.
for fruit in fruits:
    print(fruit)
# Output: apple banana cherry (each on a new line)

# 2. Loop through a range of numbers
for i in range(1, 6):
    print(i)
# Output: 1 2 3 4 5

# 3. Loop through a string
for letter in "hello":
    print(letter)
# Output: h e l l o (each on a new line)

# 4. Using break to exit early
for num in range(10):
    if num == 3:
        break
    print(num)
# Output: 0 1 2

# 5. Using continue to skip an iteration
for num in range(5):
    if num == 2:
        continue
    print(num)
# Output: 0 1 3 4

"""
Tips:
-----
- Use range() to loop a specific number of times.
- Use break to exit the loop early.
- Use continue to skip the rest of the current loop iteration.

"""