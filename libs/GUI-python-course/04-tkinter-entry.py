"""
04-entry.py

Beginner's guide to Tkinter Entry widgets.

Overview:
---------
An Entry widget is a single-line text box where users can type input.
It's commonly used for forms, search boxes, passwords, and any place you need user text input.

How Entry Works:
----------------
- Create an entry with tk.Entry(parent, options)
- parent: The window or frame where the entry will appear (usually 'root').
- options: Customize the entry (width, font, show, etc.).
- Use entry.get() to retrieve the text the user typed.
- Use entry.insert(index, text) to set default or initial text.
- Use entry.delete(start, end) to clear or remove text.
- Use show="*" to hide input (for passwords).

Common Entry Options:
--------------------
- width: Number of characters wide.
- font: Font family, size, and style.
- show: Character to display instead of actual text (e.g., "*").
- state: "normal" or "disabled".

Examples:
---------
"""

import tkinter as tk

def show_input():
    print("You entered:", entry.get())

root = tk.Tk()
root.title("Entry Example")

# Basic entry
entry = tk.Entry(root, width=30)
entry.pack(pady=10)

# Set default text
entry.insert(0, "Type here...")

# Password entry
password_entry = tk.Entry(root, show="*")
password_entry.pack(pady=10)
password_entry.insert(0, "secret")

button = tk.Button(root, text="Show Input", command=show_input)
button.pack(pady=10)

# Clear entry example
def clear_entry():
    entry.delete(0, tk.END)

clear_btn = tk.Button(root, text="Clear Entry", command=clear_entry)
clear_btn.pack(pady=10)

root.mainloop()

"""
Tips:
-----
- Use entry.get() to get user input.
- Use entry.insert(0, "text") to set default text.
- Use entry.delete(0, tk.END) to clear the entry.
- Use show="*" for password fields.
- Entry widgets are for single-line input; use Text for multi-line.
- Official docs: https://docs.python.org/3/library/tkinter.html#entry-widget
"""
