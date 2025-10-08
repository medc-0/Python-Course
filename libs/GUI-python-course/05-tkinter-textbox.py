"""
05-textbox.py

Beginner's guide to Tkinter Text widgets.

Overview:
---------
A Text widget is a multi-line text box for displaying or editing large amounts of text.
It's useful for notes, logs, chat windows, code editors, or any place you need multi-line input/output.

How Text Works:
---------------
- Create a text box with tk.Text(parent, options)
- parent: The window or frame where the text box will appear (usually 'root').
- options: Customize the text box (height, width, font, etc.).
- Use textbox.get(start, end) to retrieve text (e.g., "1.0" for start, tk.END for end).
- Use textbox.insert(index, text) to add text at a position.
- Use textbox.delete(start, end) to remove text.
- You can scroll with a Scrollbar for large content.

Common Text Options:
-------------------
- height: Number of lines tall.
- width: Number of characters wide.
- font: Font family, size, and style.
- wrap: "word" or "char" for line wrapping.

Examples:
---------
"""

import tkinter as tk

def show_text():
    print("Text box content:", textbox.get("1.0", tk.END))

root = tk.Tk()
root.title("Text Box Example")

# Basic text box
textbox = tk.Text(root, height=5, width=40)
textbox.pack(pady=10)

# Insert default text
textbox.insert(tk.END, "Welcome to the Text widget!\nType multiple lines here.")

button = tk.Button(root, text="Show Text", command=show_text)
button.pack(pady=10)

# Clear text box example
def clear_text():
    textbox.delete("1.0", tk.END)

clear_btn = tk.Button(root, text="Clear Text", command=clear_text)
clear_btn.pack(pady=10)

root.mainloop()

"""
Tips:
-----
- Use textbox.get("1.0", tk.END) to get all text.
- Use textbox.insert(tk.END, "text") to add text.
- Use textbox.delete("1.0", tk.END) to clear the text box.
- Add a Scrollbar for large content.
- Text widgets are for multi-line input; use Entry for single-line.
- Official docs: https://docs.python.org/3/library/tkinter.html#text-widget
"""
