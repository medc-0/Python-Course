"""
05-operators.py

Beginner's guide to Python comparison and logical operators.

Comparison Operators:
---------------------
Used to compare values. Result is always True or False.

- == : Equal to
- != : Not equal to
- >  : Greater than
- <  : Less than
- >= : Greater than or equal to
- <= : Less than or equal to
"""

# Examples:
a = 5
b = 3
print(a == b)   # is 'a' value EQUAL TO 'b' value: Output: False
print(a != b)   # Output: True
print(a > b)    # Output: True
print(a < b)    # Output: False
print(a >= 5)   # Output: True
print(b <= 3)   # Output: True

"""
Logical Operators:
------------------
Used to combine comparison statements.

- and : True if both statements are True
- or  : True if at least one statement is True
- not : Reverses the result (True becomes False, False becomes True)
"""

# Examples:
x = 10
y = 20

print(x > 5 and y < 30)   # Output: True
print(x > 15 or y == 20)  # Output: True
print(not(x == 10))       # Output: False

"""
Tips:
-----
- Use comparison operators to check conditions.
- Use logical operators to combine multiple conditions.
"""