"""
18-events.py

Events let you respond to user actions (mouse, keyboard, etc).

Overview:
---------
Bind functions to events like clicks, key presses, mouse movement.

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Events Example")

def on_click(event):
    print("Mouse clicked at:", event.x, event.y)

def on_key(event):
    print("Key pressed:", event.char)

root.bind("<Button-1>", on_click)
root.bind("<Key>", on_key)

root.mainloop()

"""
Tips:
-----
- Use root.bind("<Event>", function) to handle events.
- Common events: <Button-1> (left click), <Key> (key press), <Motion> (mouse move).
"""
