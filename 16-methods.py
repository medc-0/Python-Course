"""
16-methods.py

Showcase of 50+ commonly used Python methods for strings, lists, dictionaries, sets, and more.

Overview:
---------
A method is a function that belongs to an object (like a string, list, dict, set, etc.).
Methods are called with a dot: object.method(arguments).

String Methods:
---------------
"""

text = " Hello, Python! "

print(text.upper())        # ' HELLO, PYTHON! '
print(text.lower())        # ' hello, python! '
print(text.title())        # ' Hello, Python! '
print(text.capitalize())   # ' hello, python! '
print(text.strip())        # 'Hello, Python!'
print(text.lstrip())       # 'Hello, Python! '
print(text.rstrip())       # ' Hello, Python!'
print(text.replace("Python", "World"))  # ' Hello, World! '
print(text.find("Python")) # 8
print(text.index("Python"))# 8
print(text.count("o"))     # 2
print(text.startswith(" "))# True
print(text.endswith("! ")) # True
print(text.split(","))     # [' Hello', ' Python! ']
print(text.join(["A", "B"])) # 'A Hello, Python! B'
print(text.isalpha())      # False
print(text.isdigit())      # False
print(text.isalnum())      # False
print(text.islower())      # False
print(text.isupper())      # False
print(text.isspace())      # False
print(text.swapcase())     # ' hELLO, pYTHON! '
print(text.center(20, "-"))# '-- Hello, Python! --'
print(text.zfill(20))      # '00000 Hello, Python! '
print(text.partition(",")) # (' Hello', ',', ' Python! ')
print(text.rsplit(" ", 1)) # [' Hello, Python!', '']

# List Methods:
# -------------
numbers = [1, 2, 3, 2]

numbers.append(4)          # [1, 2, 3, 2, 4]
numbers.extend([5, 6])     # [1, 2, 3, 2, 4, 5, 6]
numbers.insert(0, 0)       # [0, 1, 2, 3, 2, 4, 5, 6]
numbers.remove(2)          # [0, 1, 3, 2, 4, 5, 6]
numbers.pop()              # removes last item
numbers.pop(0)             # removes first item
numbers.clear()            # []
numbers.count(2)           # 1
numbers.index(3)           # 2
numbers.sort()             # sorts list
numbers.reverse()          # reverses list
numbers.copy()             # returns a shallow copy

# Dictionary Methods:
# -------------------
person = {"name": "Alice", "age": 30}

person.get("name")         # 'Alice'
person.keys()              # dict_keys(['name', 'age'])
person.values()            # dict_values(['Alice', 30])
person.items()             # dict_items([('name', 'Alice'), ('age', 30)])
person.update({"city": "Paris"})
person.pop("age")          # removes 'age'
person.popitem()           # removes last item
person.clear()             # {}
person.setdefault("country", "France") # adds if not exists
person.copy()              # shallow copy

# Set Methods:
# ------------
fruits = {"apple", "banana", "cherry"}

fruits.add("orange")
fruits.remove("banana")
fruits.discard("banana")   # no error if not found
fruits.pop()               # removes random item
fruits.clear()
fruits.union({"kiwi"})
fruits.intersection({"apple", "kiwi"})
fruits.difference({"banana"})
fruits.symmetric_difference({"banana"})
fruits.update({"kiwi", "melon"})
fruits.isdisjoint({"pear"})
fruits.issubset({"apple", "banana", "cherry", "orange"})
fruits.issuperset({"apple"})

# Other Useful Built-in Methods:
# ------------------------------
# For files
f = open("file.txt", "w")
f.write("Hello")
f.close()

# For numbers
x = -5
print(abs(x))              # 5
print(round(3.14159, 2))   # 3.14

# For objects
print(dir(fruits))         # Lists all methods/attributes

"""
Tips:
-----
- Use help(object) or dir(object) to explore more methods.
- Methods make working with objects easier and more powerful.

"""