"""
13-file-dialog.py

File dialogs let users open or save files.

Overview:
---------
Use tkinter.filedialog for file selection dialogs.

Examples:
---------
"""

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.title("File Dialog Example")

def open_file():
    filename = filedialog.askopenfilename()
    print("Opened file:", filename)

def save_file():
    filename = filedialog.asksaveasfilename()
    print("Save file as:", filename)

tk.Button(root, text="Open File", command=open_file).pack(pady=5)
tk.Button(root, text="Save File", command=save_file).pack(pady=5)

root.mainloop()

"""
Tips:
-----
- Use askopenfilename for opening, asksaveasfilename for saving.
- You can filter file types with filetypes=[("Text Files", "*.txt")].
"""
