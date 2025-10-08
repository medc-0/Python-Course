"""
14-images.py

Display images in a Tkinter window.

Overview:
---------
Use PIL (Pillow) and ImageTk to show images.

Examples:
---------
"""

import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Images Example")

img = Image.open("example.png")  # Use your own image file
photo = ImageTk.PhotoImage(img)

label = tk.Label(root, image=photo)
label.pack()

root.mainloop()

"""
Tips:
-----
- Install Pillow: pip install pillow
- Keep a reference to the PhotoImage object to prevent garbage collection.
- You can resize images with img.resize().
"""
