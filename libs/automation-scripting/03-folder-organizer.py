"""
03-folder-organizer.py

Beginner's guide to organizing files by extension and using the shutil module.

Overview:
---------
This script shows how to automatically move files into folders based on their file type (extension).
It uses the `shutil` module, which is a built-in Python library for high-level file operations.

What is shutil?
---------------
- `shutil` stands for "shell utilities".
- It provides functions to copy, move, and remove files and folders.
- Common functions:
    - shutil.move(src, destination): Moves a file or folder from src to destination.
    - shutil.copy(src, destination): Copies a file from src to destination.
    - shutil.copytree(src, destination): Copies an entire folder and its contents.
    - shutil.rmtree(path): Removes a folder and all its contents.

How does it work?
-----------------
- Use shutil.move() to move files from one location to another.
- Use shutil.copy() to duplicate files.
- Use shutil.rmtree() to delete folders with files inside.

Examples:
---------
"""
# Organize files by extension (practice code below)

# Step 1. Import the needed imports.
import os # import os module
import shutil # import shutil 

# Step 2. setup folder where the magic happens.
folder = "organize_folder"
os.makedirs(folder, exist_ok=True) # with os.makedirs we create the folder.

# Create dummy files
extensions = ["txt", "jpg", "pdf"]
for ext in extensions:
    with open(os.path.join(folder, f"sample.{ext}"), "w") as f:
        f.write(f"Sample {ext} file")

# Organize files
for filename in os.listdir(folder):
    ext = filename.split(".")[-1]
    ext_folder = os.path.join(folder, ext)
    os.makedirs(ext_folder, exist_ok=True)
    shutil.move(os.path.join(folder, filename), os.path.join(ext_folder, filename))

print("Files organized:")
for ext in extensions:
    print(ext, ":", os.listdir(os.path.join(folder, ext)))

# Clean up
for ext in extensions:
    for filename in os.listdir(os.path.join(folder, ext)):
        os.remove(os.path.join(folder, ext, filename))
    os.rmdir(os.path.join(folder, ext))
os.rmdir(folder)

"""
Tips:
-----
- Use shutil.move() to move files and folders.
- Use shutil.copy() to copy files.
- Use shutil.rmtree() to delete folders with files inside.
- Use os.makedirs() with exist_ok=True to avoid errors if the folder already exists.
- Always test on dummy files/folders before running on important data.

"""
