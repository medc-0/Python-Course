"""
12-grid-layout.py

Grid layout arranges widgets in rows and columns.

Overview:
---------
Use .grid(row, column) to position widgets.

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Grid Layout Example")

for i in range(3):
    for j in range(3):
        tk.Button(root, text=f"({i},{j})").grid(row=i, column=j, padx=5, pady=5)

root.mainloop()

"""
Tips:
-----
- Use padx and pady for spacing.
- .grid() is great for forms and tables.
- Don't mix .pack() and .grid() in the same container.
"""
