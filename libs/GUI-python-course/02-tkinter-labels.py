"""
02-labels.py

Beginner's guide to Tkinter Labels.

Overview:
---------
A Label is a basic widget in Tkinter used to display text or images in your window.
Labels are useful for showing instructions, titles, results, or any static information.

How Labels Work:
----------------
- Create a label with tk.Label(parent, options)
- parent: The window or frame where the label will appear (usually 'root').
- options: Customize the label (text, font, color, etc.).
- Place the label in the window using .pack(), .grid(), or .place().

Common Label Options:
---------------------
- text: The string to display.
- font: Font family, size, and style (e.g., ("Arial", 16, "bold")).
- fg: Foreground/text color (e.g., "red").
- bg: Background color (e.g., "yellow").
- width/height: Size in characters/lines.
- image: Display an image (use PhotoImage or ImageTk.PhotoImage).

Examples:
---------
"""

import tkinter as tk

root = tk.Tk()
root.title("Labels Example")

# Basic label with text
label1 = tk.Label(root, text="Hello, Tkinter!", font=("Arial", 18))
label1.pack(pady=10)

# Label with custom colors
label2 = tk.Label(root, text="Second label", fg="blue", bg="yellow")
label2.pack(pady=10)

# Label with width and height
label3 = tk.Label(root, text="Wide Label", width=20, height=2, bg="lightgrey")
label3.pack(pady=10)

# Label with different font style
label4 = tk.Label(root, text="Bold & Italic", font=("Helvetica", 14, "bold italic"))
label4.pack(pady=10)

# Label with an image (requires a .gif or use PIL for more formats)
# img = tk.PhotoImage(file="example.gif")
# label_img = tk.Label(root, image=img)
# label_img.pack(pady=10)

root.mainloop()

"""
Tips:
-----
- Use font, fg (foreground color), bg (background color) for styling.
- Use .pack(), .grid(), or .place() to position widgets.
- Labels are for displaying information, not for user input.
- You can update label text with label.config(text="New text").
- For images, use tk.PhotoImage for .gif or PIL.ImageTk.PhotoImage for other formats.
- Official docs: https://docs.python.org/3/library/tkinter.html#label-widget
"""
