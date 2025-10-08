"""
19-advanced-widgets.py

Advanced widgets: Spinbox, Progressbar, Scale.

Overview:
---------
Tkinter has many widgets for advanced GUIs.

Examples:
---------
"""

import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Advanced Widgets Example")

spin = tk.Spinbox(root, from_=0, to=10)
spin.pack(pady=5)

progress = ttk.Progressbar(root, length=200, mode='determinate')
progress.pack(pady=5)
progress['value'] = 50

scale = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL)
scale.pack(pady=5)

root.mainloop()

"""
Tips:
-----
- Spinbox lets users pick a number.
- Progressbar shows progress of tasks.
- Scale is a slider for selecting values.
"""
