"""
17-menu.py

Menus let users navigate and trigger actions.

Overview:
---------
Use tk.Menu for dropdown menus.

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Menu Example")

def say_hello():
    print("Hello from menu!")

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Say Hello", command=say_hello)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

root.config(menu=menubar)
root.mainloop()

"""
Tips:
-----
- Use add_command for menu items.
- Use add_separator for dividing sections.
- Menus can have submenus for more options.
"""
