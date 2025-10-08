"""
19-fileHandling.py

Beginner's guide to file handling in Python.

Overview:
---------
File handling lets you read and write files on your computer.

Examples:
---------
"""

# Writing to a file (overwrites existing content)
with open("example.txt", "w") as f:
    f.write("Hello, file!")

# Reading from a file
with open("example.txt", "r") as f:
    content = f.read()
    print(content)  # Output: Hello, file!

# Appending to a file
with open("example.txt", "a") as f:
    f.write("\nMore text.")

# Reading lines from a file
with open("example.txt", "r") as f:
    lines = f.readlines()
    print(lines)  # Output: ['Hello, file!\n', 'More text.']

# Writing multiple lines
lines_to_write = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("lines.txt", "w") as f:
    f.writelines(lines_to_write)

# Reading file line by line
with open("lines.txt", "r") as f:
    for line in f:
        print(line.strip())

# Checking if a file exists before reading
import os
filename = "example.txt"
if os.path.exists(filename):
    with open(filename, "r") as f:
        print(f.read())
else:
    print("File does not exist.")

# Error handling when opening files
try:
    with open("notfound.txt", "r") as f:
        print(f.read())
except FileNotFoundError:
    print("File not found!")

# Remove a file
if os.path.exists("lines.txt"):
    os.remove("lines.txt")

"""
Tips:
-----
- Use "w" for write, "r" for read, "a" for append.
- Always close files (with open() does this automatically).
- Use try-except for error handling with files.
- Use os.path.exists() to check if a file exists.
- Use readlines() to get all lines as a list.
- Use writelines() to write a list of lines.

"""