"""
02-file-automation.py

Beginner's guide to automating file operations with Python.

Overview:
---------
Python makes it easy to automate file tasks: creating, reading, writing, appending, and deleting files.
This is useful for logging, saving results, batch processing, and cleaning up files.

Common File Methods:
--------------------
- with open(filename, mode): Opens a file. Modes: 'w' (write), 'r' (read), 'a' (append).
- f.write(text): Writes text to a file.
- f.read(): Reads the entire file as a string.
- f.readlines(): Reads all lines into a list.
- f.writelines(list): Writes a list of strings to a file.
- os.remove(filename): Deletes a file.
- os.path.exists(filename): Checks if a file exists.

Examples:
---------
"""

# Create and write to a file
with open("auto_file.txt", "w") as f:
    f.write("This file was created by Python automation!\n")

# Read the file
with open("auto_file.txt", "r") as f:
    print(f.read())

# Append to the file
with open("auto_file.txt", "a") as f:
    f.write("Appending more text.\n")

# Write multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("auto_file.txt", "a") as f:
    f.writelines(lines)

# Read lines from the file
with open("auto_file.txt", "r") as f:
    for line in f:
        print(line.strip())

# Delete the file
import os
if os.path.exists("auto_file.txt"):
    os.remove("auto_file.txt")
    print("File deleted.")

# Error handling
try:
    with open("notfound.txt", "r") as f:
        print(f.read())
except FileNotFoundError: # the 'fileNotFoundError' is good when there is a chance the file doesnt exist.
    print("File not found!")

"""
Tips:
-----
- Use 'w' for write (overwrites), 'a' for append, 'r' for read.
- Always close files (with open() does this automatically).
- Check existence before deleting files.
- Use try-except to prevent exceptions, and for better Error-Handling.
- Automate reports, logs, backups, and data processing with file operations.

"""
