"""
11-tkinter-frames.py

Tkinter Frames - Container widgets for organizing your GUI

Overview:
---------
Frames are container widgets that help organize and group other widgets. 
They're essential for creating well-structured, maintainable GUI layouts 
and can be styled and configured like other widgets.

Key Features:
- Container for other widgets
- Custom styling and colors
- Border and relief options
- Nested frame support
- Layout management

Common Use Cases:
- Grouping related widgets
- Creating layout sections
- Organizing form elements
- Building complex layouts
- Creating reusable components

Tips:
- Use frames to group related widgets
- Create logical sections in your layout
- Use nested frames for complex layouts
- Apply consistent styling across frames
- Consider accessibility and usability
"""

import tkinter as tk
from tkinter import messagebox

# Basic frame examples
def basic_frames():
    """Create basic frames"""
    root = tk.Tk()
    root.title("Basic Frames")
    root.geometry("500x400")
    
    # Main frame
    main_frame = tk.Frame(root, bg="lightgray", relief="raised", bd=2)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Left frame
    left_frame = tk.Frame(main_frame, bg="lightblue", relief="sunken", bd=1)
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_frame, text="Left Frame", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
    tk.Button(left_frame, text="Button 1", bg="white").pack(pady=5)
    tk.Button(left_frame, text="Button 2", bg="white").pack(pady=5)
    
    # Right frame
    right_frame = tk.Frame(main_frame, bg="lightgreen", relief="sunken", bd=1)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_frame, text="Right Frame", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=10)
    tk.Button(right_frame, text="Button 3", bg="white").pack(pady=5)
    tk.Button(right_frame, text="Button 4", bg="white").pack(pady=5)
    
    root.mainloop()

# Styled frames
def styled_frames():
    """Create frames with custom styling"""
    root = tk.Tk()
    root.title("Styled Frames")
    root.geometry("600x500")
    
    # Header frame
    header_frame = tk.Frame(root, bg="darkblue", height=60)
    header_frame.pack(fill="x", padx=10, pady=5)
    header_frame.pack_propagate(False)
    
    tk.Label(header_frame, text="Styled Frames Example", 
             font=("Arial", 16, "bold"), fg="white", bg="darkblue").pack(expand=True)
    
    # Content frame
    content_frame = tk.Frame(root, bg="lightgray")
    content_frame.pack(expand=True, fill="both", padx=10, pady=5)
    
    # Left column frame
    left_frame = tk.Frame(content_frame, bg="lightblue", relief="raised", bd=2)
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_frame, text="Left Column", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
    tk.Button(left_frame, text="Styled Button 1", bg="white", font=("Arial", 10)).pack(pady=5)
    tk.Button(left_frame, text="Styled Button 2", bg="white", font=("Arial", 10)).pack(pady=5)
    
    # Right column frame
    right_frame = tk.Frame(content_frame, bg="lightgreen", relief="raised", bd=2)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_frame, text="Right Column", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=10)
    tk.Button(right_frame, text="Styled Button 3", bg="white", font=("Arial", 10)).pack(pady=5)
    tk.Button(right_frame, text="Styled Button 4", bg="white", font=("Arial", 10)).pack(pady=5)
    
    # Footer frame
    footer_frame = tk.Frame(root, bg="darkgray", height=40)
    footer_frame.pack(fill="x", padx=10, pady=5)
    footer_frame.pack_propagate(False)
    
    tk.Label(footer_frame, text="Footer Frame", 
             font=("Arial", 10), fg="white", bg="darkgray").pack(expand=True)
    
    root.mainloop()

# Nested frames
def nested_frames():
    """Create nested frames"""
    root = tk.Tk()
    root.title("Nested Frames")
    root.geometry("600x500")
    
    # Main frame
    main_frame = tk.Frame(root, bg="lightgray", relief="raised", bd=2)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Top frame
    top_frame = tk.Frame(main_frame, bg="lightblue", relief="sunken", bd=1)
    top_frame.pack(fill="x", padx=5, pady=5)
    
    tk.Label(top_frame, text="Top Frame", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=5)
    
    # Middle frame
    middle_frame = tk.Frame(main_frame, bg="lightgreen", relief="sunken", bd=1)
    middle_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(middle_frame, text="Middle Frame", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=5)
    
    # Left sub-frame
    left_sub_frame = tk.Frame(middle_frame, bg="lightyellow", relief="raised", bd=1)
    left_sub_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_sub_frame, text="Left Sub-Frame", font=("Arial", 10, "bold"), bg="lightyellow").pack(pady=5)
    tk.Button(left_sub_frame, text="Button 1", bg="white").pack(pady=2)
    tk.Button(left_sub_frame, text="Button 2", bg="white").pack(pady=2)
    
    # Right sub-frame
    right_sub_frame = tk.Frame(middle_frame, bg="lightcoral", relief="raised", bd=1)
    right_sub_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_sub_frame, text="Right Sub-Frame", font=("Arial", 10, "bold"), bg="lightcoral").pack(pady=5)
    tk.Button(right_sub_frame, text="Button 3", bg="white").pack(pady=2)
    tk.Button(right_sub_frame, text="Button 4", bg="white").pack(pady=2)
    
    # Bottom frame
    bottom_frame = tk.Frame(main_frame, bg="lightpink", relief="sunken", bd=1)
    bottom_frame.pack(fill="x", padx=5, pady=5)
    
    tk.Label(bottom_frame, text="Bottom Frame", font=("Arial", 12, "bold"), bg="lightpink").pack(pady=5)
    
    root.mainloop()

# Frame with different reliefs
def relief_frames():
    """Create frames with different relief styles"""
    root = tk.Tk()
    root.title("Frame Relief Styles")
    root.geometry("600x500")
    
    # Relief styles
    reliefs = ["flat", "raised", "sunken", "ridge", "groove"]
    colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral", "lightpink"]
    
    # Create frames with different reliefs
    for i, (relief, color) in enumerate(zip(reliefs, colors)):
        frame = tk.Frame(root, bg=color, relief=relief, bd=3)
        frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame, text=f"Relief: {relief}", font=("Arial", 12, "bold"), bg=color).pack(pady=10)
        tk.Button(frame, text=f"Button {i+1}", bg="white").pack(pady=5)
    
    root.mainloop()

# Frame with dynamic content
def dynamic_frames():
    """Create frames with dynamic content"""
    root = tk.Tk()
    root.title("Dynamic Frames")
    root.geometry("600x500")
    
    # Main frame
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Control frame
    control_frame = tk.Frame(main_frame, bg="lightblue", relief="raised", bd=2)
    control_frame.pack(fill="x", padx=5, pady=5)
    
    tk.Label(control_frame, text="Dynamic Frame Controls", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=5)
    
    # Buttons
    button_frame = tk.Frame(control_frame, bg="lightblue")
    button_frame.pack(pady=5)
    
    def add_frame():
        new_frame = tk.Frame(main_frame, bg="lightgreen", relief="raised", bd=1)
        new_frame.pack(fill="x", padx=5, pady=2)
        
        tk.Label(new_frame, text=f"Dynamic Frame {len(main_frame.winfo_children())}", 
                font=("Arial", 10, "bold"), bg="lightgreen").pack(side="left", padx=5, pady=5)
        
        tk.Button(new_frame, text="Remove", command=lambda: new_frame.destroy(), 
                 bg="lightcoral").pack(side="right", padx=5, pady=5)
    
    def clear_frames():
        for child in main_frame.winfo_children():
            if isinstance(child, tk.Frame) and child != control_frame:
                child.destroy()
    
    tk.Button(button_frame, text="Add Frame", command=add_frame, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear Frames", command=clear_frames, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Frame with layout management
def layout_frames():
    """Create frames with different layout management"""
    root = tk.Tk()
    root.title("Layout Management Frames")
    root.geometry("700x600")
    
    # Pack layout frame
    pack_frame = tk.LabelFrame(root, text="Pack Layout", font=("Arial", 12, "bold"))
    pack_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(pack_frame, text="Top", bg="red", fg="white").pack(fill="x", pady=2)
    tk.Label(pack_frame, text="Middle", bg="green", fg="white").pack(fill="x", pady=2)
    tk.Label(pack_frame, text="Bottom", bg="blue", fg="white").pack(fill="x", pady=2)
    
    # Grid layout frame
    grid_frame = tk.LabelFrame(root, text="Grid Layout", font=("Arial", 12, "bold"))
    grid_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    # Grid buttons
    for i in range(3):
        for j in range(3):
            btn = tk.Button(grid_frame, text=f"{i},{j}", width=8, height=2)
            btn.grid(row=i, column=j, padx=2, pady=2)
    
    # Place layout frame
    place_frame = tk.LabelFrame(root, text="Place Layout", font=("Arial", 12, "bold"))
    place_frame.pack(fill="x", padx=5, pady=5)
    
    tk.Label(place_frame, text="Absolute", bg="yellow").place(x=50, y=20, width=100, height=30)
    tk.Label(place_frame, text="Positioning", bg="orange").place(x=200, y=20, width=100, height=30)
    
    root.mainloop()

# Complete frame example
def complete_frame_example():
    """Complete example showing all frame features"""
    root = tk.Tk()
    root.title("Complete Frame Example")
    root.geometry("800x600")
    
    # Header frame
    header_frame = tk.Frame(root, bg="darkblue", height=60)
    header_frame.pack(fill="x", padx=10, pady=5)
    header_frame.pack_propagate(False)
    
    tk.Label(header_frame, text="Complete Frame Example", 
             font=("Arial", 18, "bold"), fg="white", bg="darkblue").pack(expand=True)
    
    # Main content frame
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=10, pady=5)
    
    # Left column frame
    left_frame = tk.Frame(main_frame, bg="lightblue", relief="raised", bd=2)
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_frame, text="Left Column", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
    
    # Left sub-frames
    left_sub_frame1 = tk.Frame(left_frame, bg="lightyellow", relief="sunken", bd=1)
    left_sub_frame1.pack(fill="x", padx=5, pady=5)
    
    tk.Label(left_sub_frame1, text="Sub-Frame 1", font=("Arial", 10, "bold"), bg="lightyellow").pack(pady=5)
    tk.Button(left_sub_frame1, text="Button 1", bg="white").pack(pady=2)
    tk.Button(left_sub_frame1, text="Button 2", bg="white").pack(pady=2)
    
    left_sub_frame2 = tk.Frame(left_frame, bg="lightgreen", relief="sunken", bd=1)
    left_sub_frame2.pack(fill="x", padx=5, pady=5)
    
    tk.Label(left_sub_frame2, text="Sub-Frame 2", font=("Arial", 10, "bold"), bg="lightgreen").pack(pady=5)
    tk.Button(left_sub_frame2, text="Button 3", bg="white").pack(pady=2)
    tk.Button(left_sub_frame2, text="Button 4", bg="white").pack(pady=2)
    
    # Right column frame
    right_frame = tk.Frame(main_frame, bg="lightgreen", relief="raised", bd=2)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_frame, text="Right Column", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=10)
    
    # Right sub-frames
    right_sub_frame1 = tk.Frame(right_frame, bg="lightcoral", relief="sunken", bd=1)
    right_sub_frame1.pack(fill="x", padx=5, pady=5)
    
    tk.Label(right_sub_frame1, text="Sub-Frame 3", font=("Arial", 10, "bold"), bg="lightcoral").pack(pady=5)
    tk.Button(right_sub_frame1, text="Button 5", bg="white").pack(pady=2)
    tk.Button(right_sub_frame1, text="Button 6", bg="white").pack(pady=2)
    
    right_sub_frame2 = tk.Frame(right_frame, bg="lightpink", relief="sunken", bd=1)
    right_sub_frame2.pack(fill="x", padx=5, pady=5)
    
    tk.Label(right_sub_frame2, text="Sub-Frame 4", font=("Arial", 10, "bold"), bg="lightpink").pack(pady=5)
    tk.Button(right_sub_frame2, text="Button 7", bg="white").pack(pady=2)
    tk.Button(right_sub_frame2, text="Button 8", bg="white").pack(pady=2)
    
    # Footer frame
    footer_frame = tk.Frame(root, bg="darkgray", height=40)
    footer_frame.pack(fill="x", padx=10, pady=5)
    footer_frame.pack_propagate(False)
    
    tk.Label(footer_frame, text="Footer Frame", 
             font=("Arial", 10), fg="white", bg="darkgray").pack(expand=True)
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Frame Examples")
    print("=" * 30)
    print("1. Basic Frames")
    print("2. Styled Frames")
    print("3. Nested Frames")
    print("4. Relief Frames")
    print("5. Dynamic Frames")
    print("6. Layout Frames")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_frames()
    elif choice == "2":
        styled_frames()
    elif choice == "3":
        nested_frames()
    elif choice == "4":
        relief_frames()
    elif choice == "5":
        dynamic_frames()
    elif choice == "6":
        layout_frames()
    elif choice == "7":
        complete_frame_example()
    else:
        print("Invalid choice. Running basic frames...")
        basic_frames()

"""
Frame Properties:
------------------
- bg: Background color
- fg: Foreground color
- relief: Border style (flat, raised, sunken, ridge, groove)
- bd: Border width
- width: Frame width
- height: Frame height
- padx: Horizontal padding
- pady: Vertical padding

Common Relief Styles:
--------------------
- "flat": No border
- "raised": 3D raised appearance
- "sunken": 3D sunken appearance
- "ridge": Raised border
- "groove": Sunken border

Layout Management:
------------------
- pack(): Simple vertical/horizontal layout
- grid(): Table-like layout with rows and columns
- place(): Absolute positioning

Best Practices:
--------------
- Use frames to group related widgets
- Create logical sections in your layout
- Use nested frames for complex layouts
- Apply consistent styling across frames
- Consider accessibility and usability
- Test frame functionality thoroughly

Extra Tips:
-----------
- Use LabelFrame for grouped widgets with titles
- Use relief and borderwidth for visual separation
- Use padx and pady for proper spacing
- Use pack_propagate(False) to control frame size
- Use winfo_children() to access child widgets
- Consider using ttk.Frame for modern styling
"""