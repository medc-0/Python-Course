"""
01-tkinter-intro.py

Beginner's guide to Tkinter GUI programming.

Overview:
---------
Tkinter is Python's built-in library for creating graphical user interfaces (GUIs).
With Tkinter, you can build windows, dialogs, buttons, text boxes, and more—making your programs interactive and user-friendly.

How Tkinter Works:
------------------
- Tkinter is a wrapper around the Tcl/Tk GUI toolkit.
- It provides classes and functions to create windows and add widgets (GUI elements).
- Every Tkinter app starts by creating a main window (root).
- Widgets (like buttons, labels, entries) are added to the window.
- The mainloop keeps the window open and responsive to user actions.

Step-by-Step Explanation:
-------------------------
1. import tkinter as tk
   - Imports the Tkinter library and gives it the alias 'tk' for easier access.

2. root = tk.Tk()
   - Creates the main application window (called 'root').
   - All other widgets will be placed inside this window.

3. root.title("Tkinter Intro")
   - Sets the window's title (shown at the top of the window).

4. root.geometry("400x300")
   - Sets the window size to 400 pixels wide and 300 pixels tall.

5. root.mainloop()
   - Starts the Tkinter event loop.
   - Keeps the window open and waits for user actions (like clicks, typing).
   - Without mainloop(), the window would appear and immediately close.

Example: Create a basic window
-----------------------------
"""

import tkinter as tk  # 1. Import the Tkinter library

root = tk.Tk()        # 2. Create the main window
root.title("Tkinter Intro")      # 3. Set the window title
root.geometry("400x300")         # 4. Set the window size
root.mainloop()                  # 5. Start the event loop

"""
Tips:
-----
- Every Tkinter app starts with tk.Tk() to create the main window.
- Use root.title() to set the window title.
- Use root.geometry() to set window size.
- root.mainloop() keeps the window open and responsive.
- You can add widgets (buttons, labels, etc.) to the window after creating it.
- Tkinter is cross-platform: works on Windows, macOS, and Linux.
- Official docs: https://docs.python.org/3/library/tkinter.html
"""
