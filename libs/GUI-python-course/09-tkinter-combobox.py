"""
09-combobox.py

Comboboxes (dropdowns) let users select from a list.

Overview:
---------
Use ttk.Combobox for dropdown selection.

Examples:
---------
"""

import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Combobox Example")

combo = ttk.Combobox(root, values=["Red", "Green", "Blue"])
combo.pack(pady=10)

def show_color():
    print("Selected color:", combo.get())

button = tk.Button(root, text="Show Color", command=show_color)
button.pack(pady=10)

root.mainloop()

"""
Tips:
-----
- Combobox is in tkinter.ttk (themed widgets).
- Use combo.get() to get selected value.
- You can set combo.current(index) for default selection.
"""
