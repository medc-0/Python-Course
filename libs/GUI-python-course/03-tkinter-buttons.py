"""
03-tkinter-buttons.py

Beginner's guide to Tkinter Buttons.

Overview:
---------
A Button is a clickable widget in Tkinter that lets users trigger actions, like running a function or closing a window.
Buttons are essential for interactive GUIs.

How Buttons Work:
-----------------
- Create a button with tk.Button(parent, options)
- parent: The window or frame where the button will appear (usually 'root').
- options: Customize the button (text, color, command, etc.).
- Place the button in the window using .pack(), .grid(), or .place().
- Use the 'command' option to specify a function to run when the button is clicked.

Common Button Options:
---------------------
- text: The label shown on the button.
- command: The function to call when clicked (no parentheses).
- fg: Foreground/text color.
- bg: Background color.
- font: Font family, size, and style.
- width/height: Size in characters/lines.
- state: "normal" (enabled) or "disabled".

Examples:
---------

"""

import tkinter as tk
import pyttsx3 # pip install pyttsx3

def say_hello():
    text = "Hello World"
    engine.say(text)
    engine.runAndWait()

engine = pyttsx3.init()
root = tk.Tk()
root.title("Buttons Example")

# Basic button with text and command
button1 = tk.Button(root, text="Click Me", command=say_hello)
button1.pack(pady=10)

# Button to quit the app
button2 = tk.Button(root, text="Quit", command=root.quit)
button2.pack(pady=10)

# Button with custom colors and font
button3 = tk.Button(root, text="Styled Button", fg="white", bg="green", font=("Arial", 14, "bold"))
button3.pack(pady=10)

# Disabled button
button4 = tk.Button(root, text="Disabled", state="disabled")
button4.pack(pady=10)

# Button with width and height
button5 = tk.Button(root, text="Wide Button", width=20, height=2)
button5.pack(pady=10)

root.mainloop()

"""
Tips:
-----
- Use command= to specify a function to run when clicked (no parentheses).
- You can change button color, size, and font for better UI.
- Use .config() to update button properties (e.g., button1.config(text="New Text")).
- Use state="disabled" to disable a button.
- Buttons are the main way users interact with your GUI.
- Official docs: https://docs.python.org/3/library/tkinter.html#button-widget
"""
