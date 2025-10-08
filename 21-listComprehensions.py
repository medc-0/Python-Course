"""
21-listComprehensions.py

Beginner's guide to Python list comprehensions.

Overview:
---------
List comprehensions are a concise way to create lists in Python. They provide a more readable and efficient way to generate lists compared to traditional for loops.

What are List Comprehensions?
----------------------------
List comprehensions are a Pythonic way to create lists by applying an expression to each item in an iterable (like a list, range, or string).

Basic Syntax:
-------------
[expression for item in iterable]

Advanced Syntax:
----------------
[expression for item in iterable if condition]

Examples:
---------
"""

# 1. Basic list comprehension
# Traditional way:
squares_traditional = []
for i in range(1, 6):
    squares_traditional.append(i ** 2)
print("Traditional way:", squares_traditional)

# List comprehension way:
squares = [i ** 2 for i in range(1, 6)]
print("List comprehension:", squares)  # Output: [1, 4, 9, 16, 25]

# 2. List comprehension with strings
words = ["hello", "world", "python", "programming"]
word_lengths = [len(word) for word in words]
print("Word lengths:", word_lengths)  # Output: [5, 5, 6, 11]

# 3. List comprehension with conditions
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [num for num in numbers if num % 2 == 0]
print("Even numbers:", even_numbers)  # Output: [2, 4, 6, 8, 10]

# 4. List comprehension with transformations
names = ["alice", "bob", "charlie"]
capitalized_names = [name.capitalize() for name in names]
print("Capitalized names:", capitalized_names)  # Output: ['Alice', 'Bob', 'Charlie']

# 5. Nested list comprehensions
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print("Flattened matrix:", flattened)  # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 6. List comprehension with multiple conditions
numbers = range(1, 21)
filtered_numbers = [num for num in numbers if num % 2 == 0 and num % 3 == 0]
print("Numbers divisible by 2 and 3:", filtered_numbers)  # Output: [6, 12, 18]

# 7. List comprehension with if-else
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = ["even" if num % 2 == 0 else "odd" for num in numbers]
print("Even/Odd classification:", result)

# 8. List comprehension with functions
def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
squared = [square(num) for num in numbers]
print("Squared numbers:", squared)  # Output: [1, 4, 9, 16, 25]

# 9. List comprehension with string operations
text = "Hello World Python Programming"
words = text.split()
long_words = [word for word in words if len(word) > 5]
print("Long words:", long_words)  # Output: ['Python', 'Programming']

# 10. List comprehension with dictionaries
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]
high_grades = [student["name"] for student in students if student["grade"] >= 80]
print("Students with high grades:", high_grades)  # Output: ['Alice', 'Bob']

"""
When to Use List Comprehensions:
-------------------------------
- When you need to create a new list from an existing iterable
- When the logic is simple and fits on one line
- When you want more readable and Pythonic code
- When performance is important (they're often faster than loops)

When NOT to Use List Comprehensions:
------------------------------------
- When the logic is complex and hard to read
- When you need multiple statements in the loop
- When you need to modify existing variables
- When the comprehension becomes too long and unreadable

Tips:
-----
- Start with simple comprehensions and gradually add complexity
- Use meaningful variable names
- Don't sacrifice readability for brevity
- Practice with different data types and conditions
- Remember: [expression for item in iterable if condition]

"""
