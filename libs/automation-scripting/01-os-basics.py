"""
01-os-basics.py

Beginner's guide to the os module for automation and scripting.

Overview:
---------
The `os` module is a built-in Python library that lets you interact with your computer's operating system.
You can use it to manage files and folders, get information about your system, and automate tasks.

Common os Methods:
------------------
- os.listdir(path): Lists all files and folders in the given directory.
- os.mkdir(path): Creates a new folder (directory).
- os.rename(src, destination): Renames a file or folder.
- os.rmdir(path): Removes (deletes) a folder. The folder must be empty.
- os.remove(path): Removes (deletes) a file.
- os.getcwd(): Returns the current working directory (where your script is running).
- os.chdir(path): Changes the current working directory.
- os.path.exists(path): Checks if a file or folder exists.
- os.path.join(a, b): Joins paths together (useful for cross-platform scripts).

Examples:
---------
# List all files and folders in the current directory
print(os.listdir("."))  # Output: ['file1.txt', 'folder1', ...]

# Create a new folder
os.mkdir("test_folder")  # Creates a folder named 'test_folder'

# Rename a folder
os.rename("test_folder", "renamed_folder")  # Renames 'test_folder' to 'renamed_folder'

# Remove a folder
os.rmdir("renamed_folder")  # Deletes 'renamed_folder' (must be empty)

# Get current working directory
print(os.getcwd())  # Output: path to current folder

# Change directory
# os.chdir("..")  # Changes to parent directory

# Check if a file or folder exists
print(os.path.exists("some_file.txt"))  # Output: True or False

# Join paths
full_path = os.path.join("folder", "file.txt")
print(full_path)  # Output: folder/file.txt

Example Code:
--------------
"""

import os

# List files and folders in current directory
print(os.listdir("."))

# Create a new folder
os.mkdir("test_folder") # mkdir stands for make directory = make folder basically.

# Rename a folder
os.rename("test_folder", "renamed_folder") # rename a folder pass as first argument the folder and as the second argument pass the new name.

# Remove a folder
os.rmdir("renamed_folder") # rmdir stands for remove directory = delete folder basically.

# Get current working directory
print(os.getcwd()) # get the current directory (folder).

# Change directory
# os.chdir("..")  # Uncomment to change to parent directory

"""
Tips:
-----
- Use os for file/folder management and path operations.
- Always check if a file/folder exists before removing (os.path.exists).
- Use os.path.join for building paths, especially for cross-platform scripts.
- Automate repetitive tasks like organizing, renaming, or cleaning up files.
"""
