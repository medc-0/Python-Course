"""
13-zip-automation.py

Beginner's guide to zipping and unzipping files with zipfile.

Overview:
---------
The zipfile module lets you compress files into .zip archives and extract them.
This is useful for backups, sharing files, or organizing data.

Main Features:
--------------
- Create zip files
- Add files to zip archives
- Extract files from zip archives
- List contents of a zip file

Examples:
---------
"""

import zipfile
import os

# 1. Create a zip file and add a file
with zipfile.ZipFile("test.zip", "w") as zipf:
    with open("file.txt", "w") as f:
        f.write("Hello, zip!")
    zipf.write("file.txt")

# 2. List contents of zip file
with zipfile.ZipFile("test.zip", "r") as zipf:
    print("Zip contents:", zipf.namelist())

# 3. Extract zip file
with zipfile.ZipFile("test.zip", "r") as zipf:
    zipf.extractall("extracted")

print("Zip automation done.")

# Clean up
os.remove("file.txt")
os.remove("test.zip")
for filename in os.listdir("extracted"):
    os.remove(os.path.join("extracted", filename))
os.rmdir("extracted")

"""
Tips:
-----
- Use zipfile for backup and sharing files.
- You can add multiple files to a zip archive.
- Official docs: https://docs.python.org/3/library/zipfile.html
"""
