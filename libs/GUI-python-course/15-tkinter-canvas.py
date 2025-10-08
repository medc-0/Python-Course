"""
15-canvas.py

Canvas lets you draw shapes, lines, and graphics.

Overview:
---------
Use tk.Canvas for custom drawings and graphics.

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Canvas Example")

canvas = tk.Canvas(root, width=300, height=200, bg="white")
canvas.pack()

canvas.create_rectangle(50, 50, 150, 100, fill="red")
canvas.create_oval(180, 50, 280, 150, fill="blue")
canvas.create_line(0, 0, 300, 200, fill="green", width=3)
canvas.create_text(150, 180, text="Canvas Drawing", font=("Arial", 12))

root.mainloop()

"""
Tips:
-----
- Use create_rectangle, create_oval, create_line, create_text for drawing.
- Canvas is great for games, charts, and custom graphics.
"""
