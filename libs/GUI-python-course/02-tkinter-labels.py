"""
02-tkinter-labels.py

Tkinter Labels - Display text and images in your GUI

Overview:
---------
Labels are the simplest widgets in Tkinter. They display text or images 
and are perfect for titles, descriptions, and static information.

Key Features:
- Display text with custom fonts and colors
- Show images (PNG, GIF, etc.)
- Support for text wrapping and justification
- Can be interactive with click events

Common Use Cases:
- Window titles and headers
- Form field descriptions
- Status messages
- Image displays
- Help text and instructions

Tips:
- Use labels for static content that doesn't change
- Choose appropriate fonts and colors for readability
- Use text wrapping for long descriptions
- Consider accessibility with high contrast colors
"""

import tkinter as tk
from tkinter import messagebox

# Basic label examples
def basic_labels():
    """Create basic labels with different properties"""
    root = tk.Tk()
    root.title("Tkinter Labels")
    root.geometry("400x300")
    
    # Simple label
    label1 = tk.Label(root, text="Hello, Tkinter!")
    label1.pack(pady=10)
    
    # Label with custom font
    label2 = tk.Label(root, text="Custom Font", font=("Arial", 16, "bold"))
    label2.pack(pady=5)
    
    # Label with colors
    label3 = tk.Label(root, text="Colored Text", fg="blue", bg="yellow")
    label3.pack(pady=5)
    
    # Label with padding
    label4 = tk.Label(root, text="With Padding", padx=20, pady=10, bg="lightgray")
    label4.pack(pady=5)
    
    root.mainloop()

# Interactive labels
def interactive_labels():
    """Create labels that respond to clicks"""
    root = tk.Tk()
    root.title("Interactive Labels")
    root.geometry("400x300")
    
    # Clickable label
    def on_label_click(event):
        messagebox.showinfo("Clicked", "You clicked the label!")
    
    label = tk.Label(root, text="Click me!", font=("Arial", 14), 
                    fg="blue", cursor="hand2")
    label.pack(pady=20)
    label.bind("<Button-1>", on_label_click)
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", bg="lightgreen")
    status_label.pack(pady=10)
    
    root.mainloop()

# Text wrapping labels
def wrapping_labels():
    """Create labels with text wrapping"""
    root = tk.Tk()
    root.title("Text Wrapping Labels")
    root.geometry("500x400")
    
    # Long text with wrapping
    long_text = """This is a long text that will wrap to multiple lines. 
    Tkinter labels can automatically wrap text when you set the width parameter. 
    This is useful for displaying descriptions, help text, or any content that 
    might be longer than the available space."""
    
    label = tk.Label(root, text=long_text, wraplength=400, justify="left", 
                    font=("Arial", 12), bg="lightblue", padx=10, pady=10)
    label.pack(pady=20)
    
    root.mainloop()

# Image labels
def image_labels():
    """Create labels with images"""
    root = tk.Tk()
    root.title("Image Labels")
    root.geometry("400x300")
    
    # Text with image (if available)
    try:
        # Note: You need to have an image file for this to work
        # photo = tk.PhotoImage(file="image.png")
        # label = tk.Label(root, image=photo)
        # label.pack(pady=20)
        
        # For demo purposes, show text instead
        label = tk.Label(root, text="Image would go here\n(Place your image file)", 
                        font=("Arial", 12), bg="lightgray", width=20, height=5)
        label.pack(pady=20)
    except:
        label = tk.Label(root, text="Image not found", font=("Arial", 12))
        label.pack(pady=20)
    
    root.mainloop()

# Styled labels
def styled_labels():
    """Create labels with different styles"""
    root = tk.Tk()
    root.title("Styled Labels")
    root.geometry("500x400")
    
    # Header style
    header = tk.Label(root, text="Application Header", 
                     font=("Arial", 20, "bold"), fg="white", bg="darkblue")
    header.pack(fill="x", pady=5)
    
    # Subtitle style
    subtitle = tk.Label(root, text="Subtitle Text", 
                       font=("Arial", 14, "italic"), fg="gray")
    subtitle.pack(pady=5)
    
    # Info style
    info = tk.Label(root, text="Information text with details", 
                   font=("Arial", 10), fg="darkgreen", bg="lightgreen")
    info.pack(pady=5, padx=20)
    
    # Warning style
    warning = tk.Label(root, text="Warning: This is important!", 
                      font=("Arial", 12, "bold"), fg="red", bg="yellow")
    warning.pack(pady=5, padx=20)
    
    # Success style
    success = tk.Label(root, text="Success: Operation completed", 
                      font=("Arial", 12), fg="white", bg="green")
    success.pack(pady=5, padx=20)
    
    root.mainloop()

# Dynamic labels
def dynamic_labels():
    """Create labels that change content"""
    root = tk.Tk()
    root.title("Dynamic Labels")
    root.geometry("400x300")
    
    # Counter label
    counter = 0
    counter_label = tk.Label(root, text=f"Counter: {counter}", font=("Arial", 14))
    counter_label.pack(pady=20)
    
    def update_counter():
        nonlocal counter
        counter += 1
        counter_label.config(text=f"Counter: {counter}")
    
    # Time label
    import time
    time_label = tk.Label(root, text="", font=("Arial", 12))
    time_label.pack(pady=10)
    
    def update_time():
        current_time = time.strftime("%H:%M:%S")
        time_label.config(text=f"Current time: {current_time}")
        root.after(1000, update_time)  # Update every second
    
    # Buttons
    tk.Button(root, text="Increment Counter", command=update_counter).pack(pady=5)
    tk.Button(root, text="Start Clock", command=update_time).pack(pady=5)
    
    root.mainloop()

# Complete example
def complete_label_example():
    """Complete example showing all label features"""
    root = tk.Tk()
    root.title("Complete Label Example")
    root.geometry("600x500")
    
    # Header
    header = tk.Label(root, text="Complete Label Example", 
                     font=("Arial", 18, "bold"), fg="darkblue")
    header.pack(pady=10)
    
    # Content frame
    content_frame = tk.Frame(root)
    content_frame.pack(expand=True, fill="both", padx=20, pady=10)
    
    # Left column
    left_frame = tk.Frame(content_frame)
    left_frame.pack(side="left", fill="both", expand=True)
    
    # Right column
    right_frame = tk.Frame(content_frame)
    right_frame.pack(side="right", fill="both", expand=True)
    
    # Left column labels
    tk.Label(left_frame, text="Basic Labels:", font=("Arial", 12, "bold")).pack(anchor="w")
    tk.Label(left_frame, text="Simple text").pack(anchor="w", pady=2)
    tk.Label(left_frame, text="Bold text", font=("Arial", 10, "bold")).pack(anchor="w", pady=2)
    tk.Label(left_frame, text="Colored text", fg="red").pack(anchor="w", pady=2)
    
    # Right column labels
    tk.Label(right_frame, text="Styled Labels:", font=("Arial", 12, "bold")).pack(anchor="w")
    tk.Label(right_frame, text="Success", fg="white", bg="green", padx=10).pack(anchor="w", pady=2)
    tk.Label(right_frame, text="Warning", fg="black", bg="yellow", padx=10).pack(anchor="w", pady=2)
    tk.Label(right_frame, text="Error", fg="white", bg="red", padx=10).pack(anchor="w", pady=2)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Labels Examples")
    print("=" * 30)
    print("1. Basic Labels")
    print("2. Interactive Labels")
    print("3. Wrapping Labels")
    print("4. Image Labels")
    print("5. Styled Labels")
    print("6. Dynamic Labels")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_labels()
    elif choice == "2":
        interactive_labels()
    elif choice == "3":
        wrapping_labels()
    elif choice == "4":
        image_labels()
    elif choice == "5":
        styled_labels()
    elif choice == "6":
        dynamic_labels()
    elif choice == "7":
        complete_label_example()
    else:
        print("Invalid choice. Running basic labels...")
        basic_labels()

"""
Label Properties:
-----------------
- text: The text to display
- font: Font family, size, and style
- fg: Foreground (text) color
- bg: Background color
- padx, pady: Internal padding
- width, height: Widget dimensions
- wraplength: Text wrapping width
- justify: Text alignment (left, center, right)
- cursor: Mouse cursor when hovering

Common Font Styles:
------------------
- "normal" or "": Regular text
- "bold": Bold text
- "italic": Italic text
- "bold italic": Bold and italic

Color Options:
--------------
- Named colors: "red", "blue", "green", etc.
- Hex colors: "#FF0000", "#00FF00", etc.
- RGB colors: (255, 0, 0) for red

Best Practices:
--------------
- Use clear, readable fonts
- Choose appropriate colors for accessibility
- Use text wrapping for long content
- Keep labels concise and informative
- Use consistent styling throughout your app
- Test with different screen sizes

Extra Tips:
-----------
- Labels can display both text and images
- Use relief and borderwidth for borders
- Labels can be made interactive with bind()
- Use anchor for text positioning
- Consider using ttk.Label for modern styling
"""