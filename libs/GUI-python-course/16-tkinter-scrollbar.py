"""
16-scrollbar.py

Scrollbars let users scroll through large content.

Overview:
---------
Use tk.Scrollbar with Text, Listbox, or Canvas.

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Scrollbar Example")

frame = tk.Frame(root)
frame.pack()

textbox = tk.Text(frame, height=10, width=40)
textbox.pack(side=tk.LEFT)

scrollbar = tk.Scrollbar(frame, command=textbox.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

textbox.config(yscrollcommand=scrollbar.set)

for i in range(50):
    textbox.insert(tk.END, f"Line {i+1}\n")

root.mainloop()

"""
Tips:
-----
- Use scrollbar.set and yscrollcommand for vertical scrolling.
- You can also use scrollbars with Listbox and Canvas.
"""
