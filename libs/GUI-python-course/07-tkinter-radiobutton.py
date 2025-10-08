"""
07-radiobutton.py

Radio buttons let users select one option from several.

Overview:
---------
Use tk.Radiobutton for single-choice selections.

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Radio Button Example")

var = tk.StringVar(value="A")

tk.Radiobutton(root, text="Option A", variable=var, value="A").pack()
tk.Radiobutton(root, text="Option B", variable=var, value="B").pack()
tk.Radiobutton(root, text="Option C", variable=var, value="C").pack()

def show_choice():
    print("Selected:", var.get())

button = tk.Button(root, text="Show Choice", command=show_choice)
button.pack(pady=10)

root.mainloop()

"""
Tips:
-----
- Use StringVar or IntVar for radio buttons.
- All radio buttons in a group share the same variable.
"""
