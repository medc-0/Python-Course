"""
06-clipboard-automation.py

Beginner's guide to clipboard automation with pyperclip.

Overview:
---------
The clipboard is a temporary storage area for data that the user wants to copy from one place to another.
Python's pyperclip library lets you copy and paste text to and from the clipboard automatically.

What is pyperclip?
------------------
- pyperclip is a cross-platform Python library for clipboard operations.
- It works on Windows, macOS, and Linux.
- You can use it to automate copy-paste tasks, transfer data between programs, or build productivity tools.

Main Methods:
-------------
- pyperclip.copy(text): Copies the given text to the clipboard.
- pyperclip.paste(): Returns the current text from the clipboard.

Examples:
---------
"""

import pyperclip

# 1. Copy text to clipboard
pyperclip.copy("Text to copy!")
print("Clipboard contains:", pyperclip.paste())

# 2. Copy results from a calculation
result = 2 * 5
pyperclip.copy(f"The result is {result}")
print("Clipboard now:", pyperclip.paste())

# 3. Automate copying passwords or codes
password = "Secret123!"
pyperclip.copy(password)
print("Password copied to clipboard.")

# 4. Read clipboard and process text
text = pyperclip.paste()
if "Python" in text:
    print("Clipboard contains the word 'Python'.")

# 5. Build a simple clipboard history (store last 3 copies)
history = []
for i in range(3):
    pyperclip.copy(f"Copy {i}")
    history.append(pyperclip.paste())
print("Clipboard history:", history)

"""
Tips:
-----
- Install pyperclip: pip install pyperclip
- Useful for copying results, passwords, or automating repetitive copy-paste tasks.
- You can combine pyperclip with other automation scripts for productivity.
- Works only with text (not images or files).
- Great for building clipboard managers, quick note tools, or auto-fill scripts.

"""
