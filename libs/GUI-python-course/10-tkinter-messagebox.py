"""
10-messagebox.py

Message boxes show info, warnings, or errors.

Overview:
---------
Use tkinter.messagebox for pop-up dialogs.

Examples:
---------
"""

import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.title("Messagebox Example")

def show_info():
    messagebox.showinfo("Info", "This is an info message.")

def show_warning():
    messagebox.showwarning("Warning", "This is a warning.")

def show_error():
    messagebox.showerror("Error", "This is an error.")

tk.Button(root, text="Show Info", command=show_info).pack(pady=5)
tk.Button(root, text="Show Warning", command=show_warning).pack(pady=5)
tk.Button(root, text="Show Error", command=show_error).pack(pady=5)

root.mainloop()

"""
Tips:
-----
- Use showinfo, showwarning, showerror for different dialogs.
- Messageboxes are modal (block interaction until closed).
"""
