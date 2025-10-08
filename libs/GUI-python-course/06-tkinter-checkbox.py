"""
06-checkbox.py

Checkboxes let users select options.

Overview:
---------
Use tk.Checkbutton for on/off choices.

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Checkbox Example")

var = tk.BooleanVar()
checkbox = tk.Checkbutton(root, text="I agree", variable=var)
checkbox.pack(pady=10)

def show_state():
    print("Checked:", var.get())

button = tk.Button(root, text="Check State", command=show_state)
button.pack(pady=10)

root.mainloop()

"""
Tips:
-----
- Use tk.BooleanVar() for True/False.
- Use .get() to check if checked.
- Can use IntVar for 0/1 values.
"""
