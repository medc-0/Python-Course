"""
14-if__name__main.py

Beginner's guide to `if __name__ == "__main__"` in Python.

Overview:
---------
This line checks if the script is being run directly (not imported as a module).
Code inside this block only runs when the file is executed, not when imported.

Examples:
---------
"""

def greet():
    print("Hello from greet!")

if __name__ == "__main__":
    greet()  # Output: Hello from greet!
    print("This code runs only when the file is executed directly.")

# If you import this file in another script, the code inside the if-block will NOT run.

"""
Tips:
-----
- Use this pattern to write code that should only run when the file is executed directly.
- Useful for testing or running demos.

"""