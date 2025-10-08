"""
11-tuples.py

Beginner's guide to Python tuples.

Overview:
---------
A tuple is an ordered, unchangeable (immutable) collection of items. Tuples are defined with parentheses ().

Examples:
---------
"""

# 1. Creating a tuple
fruits = ("apple", "banana", "cherry")
print(fruits)  # Output: ('apple', 'banana', 'cherry')

# 2. Accessing tuple items
print(fruits[1])  # Output: banana

# 3. Looping through a tuple
for fruit in fruits:
    print(fruit)

# 4. Tuple with mixed data types
mixed = (42, "hello", 3.14, False)
print(mixed)  # Output: (42, 'hello', 3.14, False)

# 5. Nested tuple
nested = ("apple", ("banana", "cherry"))
print(nested)  # Output: ('apple', ('banana', 'cherry'))

# 6. Single value tuple (note the comma)
single = (42,)
print(single)  # Output: (42,)

"""
Tips:
-----
- Tuples are immutable, meaning they cannot be changed after creation.
- They are defined by enclosing items in parentheses `()`.
- Use tuples when you want to store multiple items in a single variable.
- Useful for fixed collections of items, like coordinates (x, y) or RGB colors.

"""