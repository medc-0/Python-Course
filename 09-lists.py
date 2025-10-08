"""
09-lists.py

Beginner's guide to Python lists.

Overview:
---------
A list is an ordered, changeable collection of items. Lists can hold any data type and are defined with square brackets [].

Examples:
---------
"""

# Creating a list
fruits = ["apple", "banana", "cherry"]
print(fruits)  # Output: ['apple', 'banana', 'cherry']

# Accessing items
print(fruits[0])  # Output: apple

# Changing items
fruits[1] = "blueberry"
print(fruits)  # Output: ['apple', 'blueberry', 'cherry']

# Adding items
fruits.append("orange")
print(fruits)  # Output: ['apple', 'blueberry', 'cherry', 'orange']

# Removing items
fruits.remove("apple")
print(fruits)  # Output: ['blueberry', 'cherry', 'orange']

# Looping through a list
for fruit in fruits:
    print(fruit)

"""
Tips:
-----
- Lists can contain mixed types: [1, "hello", True]
- Use len(list) to get the number of items.
- Lists are mutable (can be changed).

"""