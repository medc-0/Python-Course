"""
10-sets.py

Beginner's guide to Python sets.

Overview:
---------
A set is an unordered collection of unique items. Sets are defined with curly braces {}.

Examples:
---------
"""

# Creating a set
numbers = {1, 2, 3, 2}
print(numbers)  # Output: {1, 2, 3} (duplicates removed)

# Adding items
numbers.add(4)
print(numbers)  # Output: {1, 2, 3, 4}

# Removing items
numbers.remove(2)
print(numbers)  # Output: {1, 3, 4}

# Checking membership
print(3 in numbers)  # Output: True

# Looping through a set
for num in numbers:
    print(num)

# Examples:
# ---------

# 1. Creating an empty set
my_set = set()
print(my_set)  # Output: set()

# 2. Creating a set with items
fruits = {"apple", "banana", "cherry"}
print(fruits)  # Output: {'banana', 'cherry', 'apple'}

# 3. Adding an item to a set
fruits.add("orange")
print(fruits)  # Output: {'banana', 'cherry', 'apple', 'orange'}

# 4. Trying to add a duplicate item
fruits.add("apple")
print(fruits)  # Output: {'banana', 'cherry', 'apple', 'orange'}

# 5. Removing an item
fruits.remove("banana")
print(fruits)  # Output: {'cherry', 'apple', 'orange'}

# 6. Checking if an item is in the set
print("apple" in fruits)  # Output: True
print("banana" in fruits)  # Output: False

# 7. Looping through a set
for fruit in fruits:
    print(fruit)

# 8. Set operations: Union, Intersection, Difference
setA = {1, 2, 3}
setB = {3, 4, 5}
print(setA.union(setB))         # Output: {1, 2, 3, 4, 5}
print(setA.intersection(setB))  # Output: {3}
print(setA.difference(setB))    # Output: {1, 2}

"""
Tips:
-----
- Sets do not allow duplicate items.
- Sets are unordered (no indexing).
- Useful for removing duplicates from a list.

"""