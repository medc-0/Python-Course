"""
03-tkinter-buttons.py

Tkinter Buttons - Interactive elements for user input

Overview:
---------
Buttons are essential GUI elements that allow users to interact with your 
application. They can trigger actions, submit forms, and navigate between 
different parts of your app.

Key Features:
- Click events and command callbacks
- Custom styling and colors
- Disabled/enabled states
- Keyboard shortcuts and bindings
- Image buttons and icons

Common Use Cases:
- Form submission and navigation
- Action triggers (save, delete, etc.)
- Dialog box buttons (OK, Cancel)
- Menu items and toolbar buttons
- Toggle switches and checkboxes

Tips:
- Use clear, descriptive button text
- Group related buttons together
- Provide visual feedback for actions
- Use consistent styling across your app
- Consider keyboard accessibility
"""

import tkinter as tk
from tkinter import messagebox

# Basic button examples
def basic_buttons():
    """Create basic buttons with different styles"""
    root = tk.Tk()
    root.title("Tkinter Buttons")
    root.geometry("400x300")
    
    # Simple button
    def simple_click():
        messagebox.showinfo("Button Clicked", "Simple button was clicked!")
    
    btn1 = tk.Button(root, text="Click Me!", command=simple_click)
    btn1.pack(pady=10)
    
    # Styled button
    def styled_click():
        messagebox.showinfo("Styled", "Styled button was clicked!")
    
    btn2 = tk.Button(root, text="Styled Button", command=styled_click,
                    font=("Arial", 12, "bold"), fg="white", bg="blue",
                    padx=20, pady=5)
    btn2.pack(pady=10)
    
    # Disabled button
    btn3 = tk.Button(root, text="Disabled Button", state="disabled")
    btn3.pack(pady=10)
    
    root.mainloop()

# Button with different states
def button_states():
    """Create buttons with different states"""
    root = tk.Tk()
    root.title("Button States")
    root.geometry("400x300")
    
    # Counter for demonstration
    counter = 0
    counter_label = tk.Label(root, text=f"Counter: {counter}", font=("Arial", 14))
    counter_label.pack(pady=20)
    
    def increment():
        nonlocal counter
        counter += 1
        counter_label.config(text=f"Counter: {counter}")
    
    def reset():
        nonlocal counter
        counter = 0
        counter_label.config(text=f"Counter: {counter}")
    
    # Buttons
    btn_increment = tk.Button(root, text="Increment", command=increment, bg="lightgreen")
    btn_increment.pack(pady=5)
    
    btn_reset = tk.Button(root, text="Reset", command=reset, bg="lightcoral")
    btn_reset.pack(pady=5)
    
    # Toggle button
    def toggle_button():
        if btn_toggle.cget("text") == "ON":
            btn_toggle.config(text="OFF", bg="lightgray")
        else:
            btn_toggle.config(text="ON", bg="lightgreen")
    
    btn_toggle = tk.Button(root, text="OFF", command=toggle_button, bg="lightgray")
    btn_toggle.pack(pady=5)
    
    root.mainloop()

# Button with keyboard shortcuts
def keyboard_buttons():
    """Create buttons with keyboard shortcuts"""
    root = tk.Tk()
    root.title("Keyboard Shortcuts")
    root.geometry("400x300")
    
    def save_action():
        messagebox.showinfo("Save", "File saved! (Ctrl+S)")
    
    def open_action():
        messagebox.showinfo("Open", "File opened! (Ctrl+O)")
    
    def new_action():
        messagebox.showinfo("New", "New file created! (Ctrl+N)")
    
    # Buttons with keyboard shortcuts
    btn_save = tk.Button(root, text="Save (Ctrl+S)", command=save_action, bg="lightblue")
    btn_save.pack(pady=10)
    
    btn_open = tk.Button(root, text="Open (Ctrl+O)", command=open_action, bg="lightgreen")
    btn_open.pack(pady=10)
    
    btn_new = tk.Button(root, text="New (Ctrl+N)", command=new_action, bg="lightyellow")
    btn_new.pack(pady=10)
    
    # Keyboard bindings
    root.bind("<Control-s>", lambda e: save_action())
    root.bind("<Control-o>", lambda e: open_action())
    root.bind("<Control-n>", lambda e: new_action())
    
    root.mainloop()

# Button with hover effects
def hover_buttons():
    """Create buttons with hover effects"""
    root = tk.Tk()
    root.title("Hover Effects")
    root.geometry("400x300")
    
    def on_enter(event):
        event.widget.config(bg="lightblue", fg="white")
    
    def on_leave(event):
        event.widget.config(bg="lightgray", fg="black")
    
    def button_click():
        messagebox.showinfo("Hover", "Button with hover effect!")
    
    # Button with hover effects
    btn = tk.Button(root, text="Hover Over Me!", command=button_click, 
                   bg="lightgray", font=("Arial", 12))
    btn.pack(pady=20)
    
    # Bind hover events
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    
    root.mainloop()

# Button grid layout
def button_grid():
    """Create a grid of buttons"""
    root = tk.Tk()
    root.title("Button Grid")
    root.geometry("400x400")
    
    # Calculator-style button grid
    buttons = [
        ["7", "8", "9", "/"],
        ["4", "5", "6", "*"],
        ["1", "2", "3", "-"],
        ["0", ".", "=", "+"]
    ]
    
    def button_click(value):
        messagebox.showinfo("Button", f"You clicked: {value}")
    
    # Create button grid
    for i, row in enumerate(buttons):
        for j, value in enumerate(row):
            btn = tk.Button(root, text=value, command=lambda v=value: button_click(v),
                          width=8, height=2, font=("Arial", 12))
            btn.grid(row=i, column=j, padx=2, pady=2)
    
    root.mainloop()

# Button with images
def image_buttons():
    """Create buttons with images"""
    root = tk.Tk()
    root.title("Image Buttons")
    root.geometry("400x300")
    
    def button_click(action):
        messagebox.showinfo("Action", f"Performed: {action}")
    
    # Text buttons (since we don't have image files)
    btn_save = tk.Button(root, text="💾 Save", command=lambda: button_click("Save"),
                        font=("Arial", 12), bg="lightblue")
    btn_save.pack(pady=10)
    
    btn_open = tk.Button(root, text="📁 Open", command=lambda: button_click("Open"),
                        font=("Arial", 12), bg="lightgreen")
    btn_open.pack(pady=10)
    
    btn_delete = tk.Button(root, text="🗑️ Delete", command=lambda: button_click("Delete"),
                          font=("Arial", 12), bg="lightcoral")
    btn_delete.pack(pady=10)
    
    root.mainloop()

# Complete button example
def complete_button_example():
    """Complete example showing all button features"""
    root = tk.Tk()
    root.title("Complete Button Example")
    root.geometry("500x400")
    
    # Header
    header = tk.Label(root, text="Complete Button Example", 
                     font=("Arial", 16, "bold"))
    header.pack(pady=10)
    
    # Main content frame
    content_frame = tk.Frame(root)
    content_frame.pack(expand=True, fill="both", padx=20, pady=10)
    
    # Left column
    left_frame = tk.Frame(content_frame)
    left_frame.pack(side="left", fill="both", expand=True)
    
    # Right column
    right_frame = tk.Frame(content_frame)
    right_frame.pack(side="right", fill="both", expand=True)
    
    # Left column buttons
    tk.Label(left_frame, text="Action Buttons:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    tk.Button(left_frame, text="Save", bg="lightblue", width=15).pack(pady=2, anchor="w")
    tk.Button(left_frame, text="Open", bg="lightgreen", width=15).pack(pady=2, anchor="w")
    tk.Button(left_frame, text="Delete", bg="lightcoral", width=15).pack(pady=2, anchor="w")
    
    # Right column buttons
    tk.Label(right_frame, text="Style Buttons:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    tk.Button(right_frame, text="Bold", font=("Arial", 10, "bold"), width=15).pack(pady=2, anchor="w")
    tk.Button(right_frame, text="Italic", font=("Arial", 10, "italic"), width=15).pack(pady=2, anchor="w")
    tk.Button(right_frame, text="Underline", font=("Arial", 10, "underline"), width=15).pack(pady=2, anchor="w")
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Buttons Examples")
    print("=" * 30)
    print("1. Basic Buttons")
    print("2. Button States")
    print("3. Keyboard Shortcuts")
    print("4. Hover Effects")
    print("5. Button Grid")
    print("6. Image Buttons")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_buttons()
    elif choice == "2":
        button_states()
    elif choice == "3":
        keyboard_buttons()
    elif choice == "4":
        hover_buttons()
    elif choice == "5":
        button_grid()
    elif choice == "6":
        image_buttons()
    elif choice == "7":
        complete_button_example()
    else:
        print("Invalid choice. Running basic buttons...")
        basic_buttons()

"""
Button Properties:
------------------
- text: Button text
- command: Function to call when clicked
- font: Font family, size, and style
- fg: Foreground (text) color
- bg: Background color
- padx, pady: Internal padding
- width, height: Button dimensions
- state: "normal", "disabled", or "active"
- relief: Button border style
- cursor: Mouse cursor when hovering

Button States:
--------------
- "normal": Default state, clickable
- "disabled": Grayed out, not clickable
- "active": Currently being pressed

Common Relief Styles:
--------------------
- "raised": 3D raised appearance
- "sunken": 3D sunken appearance
- "flat": Flat appearance
- "ridge": Raised border
- "groove": Sunken border

Best Practices:
--------------
- Use clear, descriptive button text
- Group related buttons together
- Provide visual feedback for actions
- Use consistent styling throughout your app
- Consider keyboard accessibility
- Test button functionality thoroughly

Extra Tips:
-----------
- Buttons can have images and text
- Use relief and borderwidth for custom borders
- Buttons can be made interactive with bind()
- Use anchor for text positioning
- Consider using ttk.Button for modern styling
- Use compound for image and text positioning
"""