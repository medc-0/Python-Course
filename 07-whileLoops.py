"""
07-whileLoops.py

Beginner's guide to Python while loops.

Overview:
---------
A while loop repeats a block of code as long as a condition is True. It's useful when you don't know in advance how many times you need to repeat.

Syntax:
-------
while condition:
    # code to run while condition is True

Examples:
---------
"""

# 1. Basic while loop
count = 1
while count <= 5:
    print(count)
    count += 1
# Output: 1 2 3 4 5 (each on a new line)

# 2. Using break to exit loop early
i = 0
while True:
    print(i)
    i += 1
    if i == 3:
        break
# Output: 0 1 2

# 3. Using continue to skip an iteration
num = 0
while num < 5:
    num += 1
    if num == 3:
        continue
    print(num)
# Output: 1 2 4 5

# 4. Loop with user input
password = ""
while password != "secret":
    password = input("Enter password: ")
print("Access granted!")

"""
Tips:
-----
- Make sure the condition will eventually become False, or the loop will run forever.
- Use break to exit the loop early.
- Use continue to skip the rest of the current loop iteration.

"""