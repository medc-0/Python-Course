"""
11-frames.py

Frames help organize widgets into sections.

Overview:
---------
Use tk.Frame to group widgets and manage layout.

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Frames Example")

frame1 = tk.Frame(root, bg="lightblue", height=100)
frame1.pack(fill=tk.BOTH, expand=True)

frame2 = tk.Frame(root, bg="lightgreen", height=100)
frame2.pack(fill=tk.BOTH, expand=True)

tk.Label(frame1, text="Frame 1").pack()
tk.Label(frame2, text="Frame 2").pack()

root.mainloop()

"""
Tips:
-----
- Use frames to organize complex GUIs.
- You can nest frames inside frames.
"""
