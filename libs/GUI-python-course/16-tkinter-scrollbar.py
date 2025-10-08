"""
16-tkinter-scrollbar.py

Tkinter Scrollbar - Scrolling functionality for widgets

Overview:
---------
Scrollbars provide scrolling functionality for widgets that can display 
more content than their visible area. They're essential for text editors, 
listboxes, and any widget that needs to handle large amounts of content.

Key Features:
- Horizontal and vertical scrolling
- Customizable appearance and behavior
- Smooth scrolling and navigation
- Keyboard and mouse support
- Integration with various widgets

Common Use Cases:
- Text editors and code editors
- Large lists and data tables
- Image viewers and galleries
- Long forms and content areas
- Navigation and browsing

Tips:
- Use scrollbars for content that exceeds widget size
- Provide both horizontal and vertical scrolling when needed
- Consider keyboard navigation for accessibility
- Use appropriate scrollbar styles for your application
- Test scrolling behavior with different content sizes
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox

# Basic scrollbar
def basic_scrollbar():
    """Create basic scrollbar"""
    root = tk.Tk()
    root.title("Basic Scrollbar")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create text widget
    text_widget = tk.Text(main_frame, height=15, width=50, wrap="none")
    text_widget.pack(side="left", fill="both", expand=True)
    
    # Create vertical scrollbar
    v_scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=text_widget.yview)
    v_scrollbar.pack(side="right", fill="y")
    
    # Create horizontal scrollbar
    h_scrollbar = tk.Scrollbar(root, orient="horizontal", command=text_widget.xview)
    h_scrollbar.pack(side="bottom", fill="x")
    
    # Configure scrollbars
    text_widget.config(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    
    # Add content
    long_text = """This is a long text that demonstrates scrolling functionality.
    
You can scroll up and down to see all the content.
The scrollbar appears automatically when needed.

This text widget supports:
- Vertical scrolling with the vertical scrollbar
- Horizontal scrolling with the horizontal scrollbar
- Mouse wheel scrolling
- Keyboard navigation (arrow keys, Page Up/Down, Home/End)

Try scrolling to see all the content!
This is a long text that demonstrates scrolling functionality.

You can scroll up and down to see all the content.
The scrollbar appears automatically when needed.

This text widget supports:
- Vertical scrolling with the vertical scrollbar
- Horizontal scrolling with the horizontal scrollbar
- Mouse wheel scrolling
- Keyboard navigation (arrow keys, Page Up/Down, Home/End)

Try scrolling to see all the content!
This is a long text that demonstrates scrolling functionality.

You can scroll up and down to see all the content.
The scrollbar appears automatically when needed.

This text widget supports:
- Vertical scrolling with the vertical scrollbar
- Horizontal scrolling with the horizontal scrollbar
- Mouse wheel scrolling
- Keyboard navigation (arrow keys, Page Up/Down, Home/End)

Try scrolling to see all the content!"""
    
    text_widget.insert("1.0", long_text)
    
    root.mainloop()

# Scrolled text widget
def scrolled_text_widget():
    """Create scrolled text widget"""
    root = tk.Tk()
    root.title("Scrolled Text Widget")
    root.geometry("500x400")
    
    # Create scrolled text widget
    scrolled_text = scrolledtext.ScrolledText(root, height=15, width=50, wrap="none")
    scrolled_text.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Add content
    long_text = """This is a ScrolledText widget that automatically includes scrollbars.

ScrolledText combines a Text widget with Scrollbar widgets for convenience.
It's perfect for applications that need to display or edit large amounts of text.

Features:
- Automatic scrollbar management
- Vertical and horizontal scrolling
- Mouse wheel support
- Keyboard navigation
- Easy to use

You can type as much text as you want, and the scrollbars will handle the overflow.
This is useful for text editors, chat applications, and log displays.

The ScrolledText widget combines a Text widget with a Scrollbar widget for convenience.
It's perfect for applications that need to display or edit large amounts of text.

Features:
- Automatic scrollbar management
- Vertical and horizontal scrolling
- Mouse wheel support
- Keyboard navigation
- Easy to use

You can type as much text as you want, and the scrollbars will handle the overflow.
This is useful for text editors, chat applications, and log displays."""
    
    scrolled_text.insert("1.0", long_text)
    
    root.mainloop()

# Listbox with scrollbar
def listbox_scrollbar():
    """Create listbox with scrollbar"""
    root = tk.Tk()
    root.title("Listbox with Scrollbar")
    root.geometry("400x300")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create listbox
    listbox = tk.Listbox(main_frame, height=15, width=40)
    listbox.pack(side="left", fill="both", expand=True)
    
    # Create scrollbar
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    
    # Configure scrollbar
    listbox.config(yscrollcommand=scrollbar.set)
    
    # Add items
    for i in range(1, 101):  # 100 items
        listbox.insert(tk.END, f"Item {i}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def add_item():
        item_count = listbox.size()
        listbox.insert(tk.END, f"Item {item_count + 1}")
    
    def clear_items():
        listbox.delete(0, tk.END)
    
    tk.Button(button_frame, text="Add Item", command=add_item, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear Items", command=clear_items, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Canvas with scrollbar
def canvas_scrollbar():
    """Create canvas with scrollbar"""
    root = tk.Tk()
    root.title("Canvas with Scrollbar")
    root.geometry("600x500")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create canvas
    canvas = tk.Canvas(main_frame, width=500, height=400, bg="white")
    canvas.pack(side="left", fill="both", expand=True)
    
    # Create vertical scrollbar
    v_scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    v_scrollbar.pack(side="right", fill="y")
    
    # Create horizontal scrollbar
    h_scrollbar = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    h_scrollbar.pack(side="bottom", fill="x")
    
    # Configure scrollbars
    canvas.config(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    
    # Set scroll region
    canvas.config(scrollregion=(0, 0, 1000, 1000))
    
    # Draw content
    def draw_content():
        canvas.delete("all")
        
        # Draw grid
        for i in range(0, 1000, 50):
            canvas.create_line(i, 0, i, 1000, fill="lightgray", width=1)
        for i in range(0, 1000, 50):
            canvas.create_line(0, i, 1000, i, fill="lightgray", width=1)
        
        # Draw shapes
        for i in range(0, 1000, 100):
            for j in range(0, 1000, 100):
                canvas.create_oval(i+10, j+10, i+40, j+40, fill="lightblue", outline="blue", width=2)
                canvas.create_text(i+25, j+25, text=f"{i},{j}", font=("Arial", 8))
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def clear_canvas():
        canvas.delete("all")
    
    tk.Button(button_frame, text="Draw Content", command=draw_content, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", command=clear_canvas, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Frame with scrollbar
def frame_scrollbar():
    """Create frame with scrollbar"""
    root = tk.Tk()
    root.title("Frame with Scrollbar")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create canvas for scrolling
    canvas = tk.Canvas(main_frame, bg="lightgray")
    canvas.pack(side="left", fill="both", expand=True)
    
    # Create scrollbar
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    
    # Configure scrollbar
    canvas.config(yscrollcommand=scrollbar.set)
    
    # Create scrollable frame
    scrollable_frame = tk.Frame(canvas, bg="lightgray")
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    
    # Add content to scrollable frame
    for i in range(20):
        frame = tk.Frame(scrollable_frame, bg="lightblue", relief="raised", bd=2)
        frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame, text=f"Frame {i+1}", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
        tk.Button(frame, text=f"Button {i+1}", bg="white").pack(pady=5)
    
    # Update scroll region
    scrollable_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
    
    root.mainloop()

# Custom scrollbar
def custom_scrollbar():
    """Create custom scrollbar"""
    root = tk.Tk()
    root.title("Custom Scrollbar")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create text widget
    text_widget = tk.Text(main_frame, height=15, width=50, wrap="none")
    text_widget.pack(side="left", fill="both", expand=True)
    
    # Create custom scrollbar
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    
    # Configure scrollbar
    text_widget.config(yscrollcommand=scrollbar.set)
    
    # Customize scrollbar appearance
    scrollbar.config(
        bg="lightblue",
        activebackground="blue",
        troughcolor="lightgray",
        width=20
    )
    
    # Add content
    long_text = """This is a custom scrollbar with different colors and appearance.

The scrollbar has been customized with:
- Light blue background
- Blue active background
- Light gray trough color
- Increased width

You can customize scrollbars to match your application's theme.
This is a custom scrollbar with different colors and appearance.

The scrollbar has been customized with:
- Light blue background
- Blue active background
- Light gray trough color
- Increased width

You can customize scrollbars to match your application's theme."""
    
    text_widget.insert("1.0", long_text)
    
    root.mainloop()

# Scrollbar with events
def scrollbar_events():
    """Create scrollbar with event handling"""
    root = tk.Tk()
    root.title("Scrollbar Events")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create text widget
    text_widget = tk.Text(main_frame, height=15, width=50, wrap="none")
    text_widget.pack(side="left", fill="both", expand=True)
    
    # Create scrollbar
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    
    # Configure scrollbar
    text_widget.config(yscrollcommand=scrollbar.set)
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 10))
    status_label.pack(pady=5)
    
    # Event handlers
    def on_scroll(*args):
        status_label.config(text=f"Scrolling: {args}")
    
    def on_scroll_end(*args):
        status_label.config(text="Scroll ended")
    
    # Bind scroll events
    scrollbar.config(command=lambda *args: (text_widget.yview(*args), on_scroll(*args)))
    
    # Add content
    long_text = """This scrollbar demonstrates event handling.

When you scroll, the status label will show the scroll events.
This is useful for implementing custom scroll behavior.

Try scrolling to see the events in action!
This scrollbar demonstrates event handling.

When you scroll, the status label will show the scroll events.
This is useful for implementing custom scroll behavior.

Try scrolling to see the events in action!"""
    
    text_widget.insert("1.0", long_text)
    
    root.mainloop()

# Complete scrollbar example
def complete_scrollbar_example():
    """Complete example showing all scrollbar features"""
    root = tk.Tk()
    root.title("Complete Scrollbar Example")
    root.geometry("800x600")
    
    # Header
    header = tk.Label(root, text="Complete Scrollbar Example", 
                     font=("Arial", 16, "bold"), bg="darkblue", fg="white")
    header.pack(fill="x", padx=10, pady=10)
    
    # Main content frame
    content_frame = tk.Frame(root, bg="lightgray")
    content_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Left column
    left_frame = tk.Frame(content_frame, bg="lightblue", relief="raised", bd=2)
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_frame, text="Text with Scrollbar", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
    
    # Create text widget with scrollbar
    text_frame = tk.Frame(left_frame)
    text_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    text_widget = tk.Text(text_frame, height=10, width=30, wrap="none")
    text_widget.pack(side="left", fill="both", expand=True)
    
    text_scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_scrollbar.pack(side="right", fill="y")
    
    text_widget.config(yscrollcommand=text_scrollbar.set)
    
    # Add content
    text_content = """This is a text widget with a scrollbar.

You can scroll to see all the content.
The scrollbar appears automatically when needed.

This demonstrates:
- Vertical scrolling
- Text editing
- Scrollbar integration
- Content overflow handling"""
    
    text_widget.insert("1.0", text_content)
    
    # Right column
    right_frame = tk.Frame(content_frame, bg="lightgreen", relief="raised", bd=2)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_frame, text="List with Scrollbar", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=10)
    
    # Create listbox with scrollbar
    list_frame = tk.Frame(right_frame)
    list_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    listbox = tk.Listbox(list_frame, height=10, width=30)
    listbox.pack(side="left", fill="both", expand=True)
    
    list_scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
    list_scrollbar.pack(side="right", fill="y")
    
    listbox.config(yscrollcommand=list_scrollbar.set)
    
    # Add items
    for i in range(1, 51):  # 50 items
        listbox.insert(tk.END, f"Item {i}")
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Scrollbar Examples")
    print("=" * 30)
    print("1. Basic Scrollbar")
    print("2. Scrolled Text Widget")
    print("3. Listbox with Scrollbar")
    print("4. Canvas with Scrollbar")
    print("5. Frame with Scrollbar")
    print("6. Custom Scrollbar")
    print("7. Scrollbar Events")
    print("8. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-8): ")
    
    if choice == "1":
        basic_scrollbar()
    elif choice == "2":
        scrolled_text_widget()
    elif choice == "3":
        listbox_scrollbar()
    elif choice == "4":
        canvas_scrollbar()
    elif choice == "5":
        frame_scrollbar()
    elif choice == "6":
        custom_scrollbar()
    elif choice == "7":
        scrollbar_events()
    elif choice == "8":
        complete_scrollbar_example()
    else:
        print("Invalid choice. Running basic scrollbar...")
        basic_scrollbar()

"""
Scrollbar Properties:
---------------------
- orient: Orientation ("vertical" or "horizontal")
- command: Function to call when scrollbar moves
- bg: Background color
- activebackground: Active background color
- troughcolor: Trough color
- width: Scrollbar width
- relief: Border style
- bd: Border width

Scrollbar Methods:
------------------
- set(): Set scrollbar position
- get(): Get scrollbar position
- configure(): Configure scrollbar properties
- cget(): Get scrollbar property value

Common Use Cases:
-----------------
- Text editors and code editors
- Large lists and data tables
- Image viewers and galleries
- Long forms and content areas
- Navigation and browsing

Best Practices:
--------------
- Use scrollbars for content that exceeds widget size
- Provide both horizontal and vertical scrolling when needed
- Consider keyboard navigation for accessibility
- Use appropriate scrollbar styles for your application
- Test scrolling behavior with different content sizes
- Use ScrolledText for automatic scrollbar management

Extra Tips:
-----------
- Use ScrolledText for automatic scrollbar management
- Use scrollregion for canvas scrolling
- Use update_idletasks() to update scroll region
- Use bbox() to get widget bounds
- Consider using ttk.Scrollbar for modern styling
- Use mouse wheel events for better user experience
"""