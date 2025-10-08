"""
14-mouse-keyboard.py

Beginner's guide to mouse and keyboard automation with pyautogui.

Overview:
---------
PyAutoGUI lets you control the mouse and keyboard with Python.
You can automate clicks, typing, moving the mouse, and more. Useful for GUI automation, testing, and repetitive tasks.

Main Features:
--------------
- Move mouse to coordinates
- Click mouse buttons
- Type text automatically
- Take screenshots
- Locate images on screen

Examples:
---------
"""

import pyautogui

# 1. Move mouse to position (100, 100)
pyautogui.moveTo(100, 100)

# 2. Click mouse
# pyautogui.click()

# 3. Type text
# pyautogui.write("Hello, world!")

# 4. Take a screenshot
# screenshot = pyautogui.screenshot()
# screenshot.save("screenshot.png")

print("Mouse and keyboard automation example.")

"""
Tips:
-----
- Install pyautogui: pip install pyautogui
- Useful for GUI automation and testing.
- Official docs: https://pyautogui.readthedocs.io/en/latest/
- Be careful: pyautogui can control your computer!
"""
