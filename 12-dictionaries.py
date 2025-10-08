"""
12-dictionaries.py

Beginner's guide to Python dictionaries.

Overview:
---------
A dictionary is an unordered collection of key-value pairs. Dictionaries are defined with curly braces {}.

Examples:
---------
"""

# Creating a dictionary
person = {"name": "Alice", "age": 30}
print(person)  # Output: {'name': 'Alice', 'age': 30}

# Accessing values
print(person["name"])  # Output: Alice

# Adding a key-value pair
person["city"] = "Paris"
print(person)  # Output: {'name': 'Alice', 'age': 30, 'city': 'Paris'}

# Removing a key-value pair
del person["age"]
print(person)  # Output: {'name': 'Alice', 'city': 'Paris'}

# Looping through keys and values
for key, value in person.items():
    print(key, value)

"""
Tips:
-----
- Keys must be unique and immutable (e.g., strings, numbers).
- Values can be any data type.
- Use dictionaries for structured data.

"""