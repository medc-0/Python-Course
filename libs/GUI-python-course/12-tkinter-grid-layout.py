"""
12-tkinter-grid-layout.py

Tkinter Grid Layout - Table-like widget positioning

Overview:
---------
Grid layout is a powerful positioning system that arranges widgets in 
rows and columns, similar to a table. It's perfect for creating 
organized, structured layouts with precise control over widget placement.

Key Features:
- Row and column-based positioning
- Spanning multiple rows/columns
- Sticky positioning for widget expansion
- Padding and spacing control
- Weight-based resizing

Common Use Cases:
- Form layouts with labels and inputs
- Data tables and spreadsheets
- Calculator interfaces
- Dashboard layouts
- Complex widget arrangements

Tips:
- Use grid for structured layouts
- Plan your row/column structure first
- Use sticky for widget expansion
- Apply consistent padding and spacing
- Test with different window sizes
"""

import tkinter as tk
from tkinter import messagebox

# Basic grid layout
def basic_grid():
    """Create basic grid layout"""
    root = tk.Tk()
    root.title("Basic Grid Layout")
    root.geometry("400x300")
    
    # Create widgets
    tk.Label(root, text="Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, width=30).grid(row=0, column=1, padx=5, pady=5)
    
    tk.Label(root, text="Email:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, width=30).grid(row=1, column=1, padx=5, pady=5)
    
    tk.Label(root, text="Age:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, width=30).grid(row=2, column=1, padx=5, pady=5)
    
    # Buttons
    tk.Button(root, text="Submit", bg="lightblue").grid(row=3, column=0, padx=5, pady=10)
    tk.Button(root, text="Cancel", bg="lightcoral").grid(row=3, column=1, padx=5, pady=10)
    
    root.mainloop()

# Grid with spanning
def spanning_grid():
    """Create grid layout with spanning"""
    root = tk.Tk()
    root.title("Grid Layout with Spanning")
    root.geometry("500x400")
    
    # Header spanning full width
    tk.Label(root, text="Header", bg="lightblue", font=("Arial", 14, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    
    # Left column
    tk.Label(root, text="Left Column", bg="lightgreen").grid(
        row=1, column=0, sticky="nsew", padx=5, pady=5)
    
    # Middle column
    tk.Label(root, text="Middle Column", bg="lightyellow").grid(
        row=1, column=1, sticky="nsew", padx=5, pady=5)
    
    # Right column
    tk.Label(root, text="Right Column", bg="lightcoral").grid(
        row=1, column=2, sticky="nsew", padx=5, pady=5)
    
    # Footer spanning full width
    tk.Label(root, text="Footer", bg="lightgray", font=("Arial", 12, "bold")).grid(
        row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    
    # Configure column weights
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)
    
    root.mainloop()

# Grid with sticky positioning
def sticky_grid():
    """Create grid layout with sticky positioning"""
    root = tk.Tk()
    root.title("Grid Layout with Sticky Positioning")
    root.geometry("500x400")
    
    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=2)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    
    # Top-left widget
    tk.Label(root, text="Top Left", bg="lightblue", font=("Arial", 12, "bold")).grid(
        row=0, column=0, sticky="nsew", padx=5, pady=5)
    
    # Top-right widget
    tk.Label(root, text="Top Right", bg="lightgreen", font=("Arial", 12, "bold")).grid(
        row=0, column=1, sticky="nsew", padx=5, pady=5)
    
    # Bottom-left widget
    tk.Label(root, text="Bottom Left", bg="lightyellow", font=("Arial", 12, "bold")).grid(
        row=1, column=0, sticky="nsew", padx=5, pady=5)
    
    # Bottom-right widget
    tk.Label(root, text="Bottom Right", bg="lightcoral", font=("Arial", 12, "bold")).grid(
        row=1, column=1, sticky="nsew", padx=5, pady=5)
    
    root.mainloop()

# Grid with padding
def padding_grid():
    """Create grid layout with padding"""
    root = tk.Tk()
    root.title("Grid Layout with Padding")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root, bg="lightgray", relief="raised", bd=2)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Configure grid weights
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)
    
    # Widgets with different padding
    tk.Label(main_frame, text="No Padding", bg="lightblue").grid(
        row=0, column=0, sticky="nsew")
    
    tk.Label(main_frame, text="Small Padding", bg="lightgreen").grid(
        row=0, column=1, sticky="nsew", padx=5, pady=5)
    
    tk.Label(main_frame, text="Large Padding", bg="lightyellow").grid(
        row=1, column=0, sticky="nsew", padx=20, pady=20)
    
    tk.Label(main_frame, text="Mixed Padding", bg="lightcoral").grid(
        row=1, column=1, sticky="nsew", padx=(10, 30), pady=(5, 15))
    
    root.mainloop()

# Grid with form layout
def form_grid():
    """Create grid layout for forms"""
    root = tk.Tk()
    root.title("Form Grid Layout")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root, bg="white")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Configure grid weights
    main_frame.columnconfigure(1, weight=1)
    
    # Form fields
    fields = [
        ("Name:", "text"),
        ("Email:", "email"),
        ("Phone:", "phone"),
        ("Address:", "text"),
        ("City:", "text"),
        ("State:", "text"),
        ("Zip:", "text")
    ]
    
    entries = {}
    
    for i, (label_text, field_type) in enumerate(fields):
        # Label
        tk.Label(main_frame, text=label_text, font=("Arial", 10, "bold")).grid(
            row=i, column=0, sticky="w", padx=(0, 10), pady=5)
        
        # Entry
        entry = tk.Entry(main_frame, width=30, font=("Arial", 10))
        entry.grid(row=i, column=1, sticky="ew", padx=(0, 0), pady=5)
        entries[field_type] = entry
    
    # Buttons
    button_frame = tk.Frame(main_frame)
    button_frame.grid(row=len(fields), column=0, columnspan=2, pady=20)
    
    tk.Button(button_frame, text="Submit", bg="lightblue", width=10).pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", bg="lightcoral", width=10).pack(side="left", padx=5)
    tk.Button(button_frame, text="Cancel", bg="lightgray", width=10).pack(side="left", padx=5)
    
    root.mainloop()

# Grid with calculator layout
def calculator_grid():
    """Create grid layout for calculator"""
    root = tk.Tk()
    root.title("Calculator Grid Layout")
    root.geometry("300x400")
    
    # Create main frame
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Configure grid weights
    for i in range(4):
        main_frame.columnconfigure(i, weight=1)
    for i in range(6):
        main_frame.rowconfigure(i, weight=1)
    
    # Display
    display = tk.Entry(main_frame, font=("Arial", 16), justify="right", state="readonly")
    display.grid(row=0, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
    
    # Calculator buttons
    buttons = [
        ("C", 1, 0), ("CE", 1, 1), ("⌫", 1, 2), ("÷", 1, 3),
        ("7", 2, 0), ("8", 2, 1), ("9", 2, 2), ("×", 2, 3),
        ("4", 3, 0), ("5", 3, 1), ("6", 3, 2), ("-", 3, 3),
        ("1", 4, 0), ("2", 4, 1), ("3", 4, 2), ("+", 4, 3),
        ("0", 5, 0), (".", 5, 1), ("=", 5, 2), ("=", 5, 3)
    ]
    
    for text, row, col in buttons:
        if text == "=":
            tk.Button(main_frame, text=text, bg="lightblue", font=("Arial", 12, "bold")).grid(
                row=row, column=col, sticky="nsew", padx=2, pady=2)
        elif text in ["÷", "×", "-", "+"]:
            tk.Button(main_frame, text=text, bg="lightgreen", font=("Arial", 12, "bold")).grid(
                row=row, column=col, sticky="nsew", padx=2, pady=2)
        elif text in ["C", "CE", "⌫"]:
            tk.Button(main_frame, text=text, bg="lightcoral", font=("Arial", 12, "bold")).grid(
                row=row, column=col, sticky="nsew", padx=2, pady=2)
        else:
            tk.Button(main_frame, text=text, bg="white", font=("Arial", 12, "bold")).grid(
                row=row, column=col, sticky="nsew", padx=2, pady=2)
    
    root.mainloop()

# Grid with dynamic content
def dynamic_grid():
    """Create grid layout with dynamic content"""
    root = tk.Tk()
    root.title("Dynamic Grid Layout")
    root.geometry("600x500")
    
    # Create main frame
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Control frame
    control_frame = tk.Frame(main_frame, bg="lightblue", relief="raised", bd=2)
    control_frame.pack(fill="x", pady=(0, 10))
    
    tk.Label(control_frame, text="Dynamic Grid Controls", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=5)
    
    # Buttons
    button_frame = tk.Frame(control_frame, bg="lightblue")
    button_frame.pack(pady=5)
    
    def add_widget():
        row = len(main_frame.grid_slaves()) // 3
        col = len(main_frame.grid_slaves()) % 3
        
        widget = tk.Label(main_frame, text=f"Widget {row},{col}", bg="lightgreen", 
                         font=("Arial", 10, "bold"))
        widget.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
        
        # Configure grid weights
        main_frame.columnconfigure(col, weight=1)
        main_frame.rowconfigure(row, weight=1)
    
    def clear_widgets():
        for widget in main_frame.grid_slaves():
            if isinstance(widget, tk.Label):
                widget.destroy()
    
    tk.Button(button_frame, text="Add Widget", command=add_widget, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear Widgets", command=clear_widgets, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Complete grid example
def complete_grid_example():
    """Complete example showing all grid layout features"""
    root = tk.Tk()
    root.title("Complete Grid Layout Example")
    root.geometry("800x600")
    
    # Header
    header = tk.Label(root, text="Complete Grid Layout Example", 
                     font=("Arial", 16, "bold"), bg="darkblue", fg="white")
    header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
    
    # Main content frame
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
    
    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)
    root.rowconfigure(1, weight=1)
    
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.columnconfigure(2, weight=1)
    main_frame.rowconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)
    
    # Left column
    tk.Label(main_frame, text="Left Column", bg="lightblue", font=("Arial", 12, "bold")).grid(
        row=0, column=0, sticky="nsew", padx=5, pady=5)
    
    # Middle column
    tk.Label(main_frame, text="Middle Column", bg="lightgreen", font=("Arial", 12, "bold")).grid(
        row=0, column=1, sticky="nsew", padx=5, pady=5)
    
    # Right column
    tk.Label(main_frame, text="Right Column", bg="lightyellow", font=("Arial", 12, "bold")).grid(
        row=0, column=2, sticky="nsew", padx=5, pady=5)
    
    # Bottom row spanning full width
    tk.Label(main_frame, text="Bottom Row", bg="lightcoral", font=("Arial", 12, "bold")).grid(
        row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    
    # Footer
    footer = tk.Label(root, text="Footer", bg="darkgray", fg="white", font=("Arial", 10))
    footer.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Grid Layout Examples")
    print("=" * 35)
    print("1. Basic Grid")
    print("2. Spanning Grid")
    print("3. Sticky Grid")
    print("4. Padding Grid")
    print("5. Form Grid")
    print("6. Calculator Grid")
    print("7. Dynamic Grid")
    print("8. Complete Example")
    print("=" * 35)
    
    choice = input("Enter your choice (1-8): ")
    
    if choice == "1":
        basic_grid()
    elif choice == "2":
        spanning_grid()
    elif choice == "3":
        sticky_grid()
    elif choice == "4":
        padding_grid()
    elif choice == "5":
        form_grid()
    elif choice == "6":
        calculator_grid()
    elif choice == "7":
        dynamic_grid()
    elif choice == "8":
        complete_grid_example()
    else:
        print("Invalid choice. Running basic grid...")
        basic_grid()

"""
Grid Layout Properties:
-----------------------
- row: Row position (0-based)
- column: Column position (0-based)
- rowspan: Number of rows to span
- columnspan: Number of columns to span
- sticky: Widget expansion (n, s, e, w, nsew)
- padx: Horizontal padding
- pady: Vertical padding
- ipadx: Internal horizontal padding
- ipady: Internal vertical padding

Sticky Values:
--------------
- "n": North (top)
- "s": South (bottom)
- "e": East (right)
- "w": West (left)
- "nsew": All directions (fill)
- "ns": North-South (vertical fill)
- "ew": East-West (horizontal fill)

Grid Configuration:
--------------------
- columnconfigure(): Configure column properties
- rowconfigure(): Configure row properties
- weight: Resize weight for column/row
- minsize: Minimum size for column/row

Best Practices:
--------------
- Use grid for structured layouts
- Plan your row/column structure first
- Use sticky for widget expansion
- Apply consistent padding and spacing
- Test with different window sizes
- Use columnconfigure and rowconfigure for resizing

Extra Tips:
-----------
- Use grid_slaves() to access grid children
- Use grid_info() to get grid information
- Use grid_remove() to hide widgets
- Use grid_forget() to remove widgets
- Use grid_propagate() to control automatic sizing
- Consider using ttk widgets for modern styling
"""