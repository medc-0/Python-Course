"""
19-tkinter-advanced-widgets.py

Tkinter Advanced Widgets - Complex and specialized widgets

Overview:
---------
Advanced widgets provide specialized functionality for complex applications. 
They include progress bars, scales, spinboxes, and other sophisticated 
controls that enhance user experience and application capabilities.

Key Features:
- Progress bars for task monitoring
- Scales for value selection
- Spinboxes for numeric input
- Notebooks for tabbed interfaces
- Treeviews for hierarchical data
- Custom widget creation

Common Use Cases:
- Progress tracking and status updates
- Value selection and configuration
- Tabbed interfaces and navigation
- Data display and organization
- Custom user interface components

Tips:
- Use appropriate widgets for specific tasks
- Provide clear labels and instructions
- Consider accessibility and usability
- Test widgets with different data types
- Use consistent styling across your application
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading

# Progress bar
def progress_bar():
    """Create progress bar example"""
    root = tk.Tk()
    root.title("Progress Bar")
    root.geometry("500x300")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Progress bar
    progress = ttk.Progressbar(main_frame, mode="determinate", length=400)
    progress.pack(pady=20)
    
    # Progress label
    progress_label = tk.Label(main_frame, text="Progress: 0%", font=("Arial", 12))
    progress_label.pack(pady=10)
    
    # Status label
    status_label = tk.Label(main_frame, text="Status: Ready", font=("Arial", 10))
    status_label.pack(pady=5)
    
    # Progress function
    def start_progress():
        progress["maximum"] = 100
        progress["value"] = 0
        
        def update_progress():
            for i in range(101):
                progress["value"] = i
                progress_label.config(text=f"Progress: {i}%")
                status_label.config(text=f"Status: Processing... {i}%")
                root.update()
                time.sleep(0.05)  # Simulate work
            
            status_label.config(text="Status: Complete!")
            messagebox.showinfo("Progress", "Task completed!")
        
        # Run in separate thread to avoid blocking UI
        threading.Thread(target=update_progress, daemon=True).start()
    
    def reset_progress():
        progress["value"] = 0
        progress_label.config(text="Progress: 0%")
        status_label.config(text="Status: Ready")
    
    # Buttons
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=20)
    
    tk.Button(button_frame, text="Start Progress", command=start_progress, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Reset", command=reset_progress, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Scale widget
def scale_widget():
    """Create scale widget example"""
    root = tk.Tk()
    root.title("Scale Widget")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Value label
    value_label = tk.Label(main_frame, text="Value: 0", font=("Arial", 14, "bold"))
    value_label.pack(pady=20)
    
    # Horizontal scale
    h_scale = tk.Scale(main_frame, from_=0, to=100, orient="horizontal", length=400,
                      command=lambda v: value_label.config(text=f"Value: {v}"))
    h_scale.pack(pady=10)
    
    # Vertical scale
    v_scale = tk.Scale(main_frame, from_=0, to=100, orient="vertical", length=200,
                      command=lambda v: value_label.config(text=f"Value: {v}"))
    v_scale.pack(side="left", padx=20)
    
    # Scale properties
    properties_frame = tk.Frame(main_frame)
    properties_frame.pack(side="right", padx=20)
    
    tk.Label(properties_frame, text="Scale Properties", font=("Arial", 12, "bold")).pack(pady=10)
    
    # Resolution
    resolution_label = tk.Label(properties_frame, text="Resolution: 1")
    resolution_label.pack(pady=5)
    
    def set_resolution(value):
        h_scale.config(resolution=float(value))
        v_scale.config(resolution=float(value))
        resolution_label.config(text=f"Resolution: {value}")
    
    resolution_scale = tk.Scale(properties_frame, from_=0.1, to=5.0, resolution=0.1, orient="horizontal",
                               command=set_resolution, length=200)
    resolution_scale.set(1.0)
    resolution_scale.pack(pady=5)
    
    # Tick interval
    tick_label = tk.Label(properties_frame, text="Tick Interval: 10")
    tick_label.pack(pady=5)
    
    def set_tick_interval(value):
        h_scale.config(tickinterval=float(value))
        v_scale.config(tickinterval=float(value))
        tick_label.config(text=f"Tick Interval: {value}")
    
    tick_scale = tk.Scale(properties_frame, from_=1, to=50, orient="horizontal",
                         command=set_tick_interval, length=200)
    tick_scale.set(10)
    tick_scale.pack(pady=5)
    
    root.mainloop()

# Spinbox widget
def spinbox_widget():
    """Create spinbox widget example"""
    root = tk.Tk()
    root.title("Spinbox Widget")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Value label
    value_label = tk.Label(main_frame, text="Value: 0", font=("Arial", 14, "bold"))
    value_label.pack(pady=20)
    
    # Basic spinbox
    basic_spinbox = tk.Spinbox(main_frame, from_=0, to=100, width=10,
                              command=lambda: value_label.config(text=f"Value: {basic_spinbox.get()}"))
    basic_spinbox.pack(pady=10)
    
    # Spinbox with values
    values_spinbox = tk.Spinbox(main_frame, values=("Red", "Green", "Blue", "Yellow", "Purple"), width=10,
                               command=lambda: value_label.config(text=f"Color: {values_spinbox.get()}"))
    values_spinbox.pack(pady=10)
    
    # Spinbox with format
    format_spinbox = tk.Spinbox(main_frame, from_=0, to=100, format="%02d", width=10,
                               command=lambda: value_label.config(text=f"Formatted: {format_spinbox.get()}"))
    format_spinbox.pack(pady=10)
    
    # Spinbox with increment
    increment_spinbox = tk.Spinbox(main_frame, from_=0, to=100, increment=5, width=10,
                                  command=lambda: value_label.config(text=f"Increment: {increment_spinbox.get()}"))
    increment_spinbox.pack(pady=10)
    
    # Spinbox with wrap
    wrap_spinbox = tk.Spinbox(main_frame, from_=0, to=10, wrap=True, width=10,
                             command=lambda: value_label.config(text=f"Wrap: {wrap_spinbox.get()}"))
    wrap_spinbox.pack(pady=10)
    
    # Buttons
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=20)
    
    def get_values():
        values = f"Basic: {basic_spinbox.get()}, Color: {values_spinbox.get()}, "
        values += f"Formatted: {format_spinbox.get()}, Increment: {increment_spinbox.get()}, "
        values += f"Wrap: {wrap_spinbox.get()}"
        messagebox.showinfo("Values", values)
    
    tk.Button(button_frame, text="Get Values", command=get_values, bg="lightblue").pack(side="left", padx=5)
    
    root.mainloop()

# Notebook widget
def notebook_widget():
    """Create notebook widget example"""
    root = tk.Tk()
    root.title("Notebook Widget")
    root.geometry("600x500")
    
    # Create notebook
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Tab 1: Text
    text_frame = ttk.Frame(notebook)
    notebook.add(text_frame, text="Text")
    
    text_widget = tk.Text(text_frame, wrap="word")
    text_widget.pack(expand=True, fill="both", padx=10, pady=10)
    
    text_content = """This is the first tab of the notebook.
    
You can add multiple tabs to organize your content.
Each tab can contain different widgets and functionality.

This text widget demonstrates:
- Multi-line text input
- Text wrapping
- Scrollable content
- Tab organization"""
    
    text_widget.insert("1.0", text_content)
    
    # Tab 2: Buttons
    button_frame = ttk.Frame(notebook)
    notebook.add(button_frame, text="Buttons")
    
    tk.Label(button_frame, text="Button Examples", font=("Arial", 14, "bold")).pack(pady=20)
    
    button_grid = tk.Frame(button_frame)
    button_grid.pack(expand=True, fill="both", padx=20, pady=20)
    
    buttons = [
        ("Button 1", "lightblue"),
        ("Button 2", "lightgreen"),
        ("Button 3", "lightcoral"),
        ("Button 4", "lightyellow"),
        ("Button 5", "lightpink"),
        ("Button 6", "lightgray")
    ]
    
    for i, (text, color) in enumerate(buttons):
        row = i // 3
        col = i % 3
        btn = tk.Button(button_grid, text=text, bg=color, width=15, height=2)
        btn.grid(row=row, column=col, padx=5, pady=5)
    
    # Tab 3: List
    list_frame = ttk.Frame(notebook)
    notebook.add(list_frame, text="List")
    
    tk.Label(list_frame, text="List Examples", font=("Arial", 14, "bold")).pack(pady=20)
    
    list_widget = tk.Listbox(list_frame, height=15)
    list_widget.pack(expand=True, fill="both", padx=20, pady=20)
    
    for i in range(1, 51):
        list_widget.insert(tk.END, f"Item {i}")
    
    # Tab 4: Tree
    tree_frame = ttk.Frame(notebook)
    notebook.add(tree_frame, text="Tree")
    
    tk.Label(tree_frame, text="Tree Examples", font=("Arial", 14, "bold")).pack(pady=20)
    
    tree = ttk.Treeview(tree_frame)
    tree.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Add tree items
    root_item = tree.insert("", "end", text="Root", open=True)
    for i in range(1, 6):
        child = tree.insert(root_item, "end", text=f"Child {i}")
        for j in range(1, 4):
            tree.insert(child, "end", text=f"Grandchild {i}.{j}")
    
    root.mainloop()

# Treeview widget
def treeview_widget():
    """Create treeview widget example"""
    root = tk.Tk()
    root.title("Treeview Widget")
    root.geometry("600x500")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Create treeview
    tree = ttk.Treeview(main_frame, columns=("size", "type"), show="tree headings")
    tree.pack(side="left", fill="both", expand=True)
    
    # Configure columns
    tree.heading("#0", text="Name")
    tree.heading("size", text="Size")
    tree.heading("type", text="Type")
    
    tree.column("#0", width=200)
    tree.column("size", width=100)
    tree.column("type", width=100)
    
    # Add data
    root_item = tree.insert("", "end", text="Documents", values=("", "Folder"), open=True)
    tree.insert(root_item, "end", text="Report.pdf", values=("2.5 MB", "PDF"))
    tree.insert(root_item, "end", text="Presentation.pptx", values=("15.2 MB", "PowerPoint"))
    tree.insert(root_item, "end", text="Spreadsheet.xlsx", values=("3.8 MB", "Excel"))
    
    images_item = tree.insert("", "end", text="Images", values=("", "Folder"), open=True)
    tree.insert(images_item, "end", text="Photo1.jpg", values=("4.2 MB", "JPEG"))
    tree.insert(images_item, "end", text="Photo2.png", values=("2.1 MB", "PNG"))
    tree.insert(images_item, "end", text="Photo3.gif", values=("1.8 MB", "GIF"))
    
    music_item = tree.insert("", "end", text="Music", values=("", "Folder"), open=True)
    tree.insert(music_item, "end", text="Song1.mp3", values=("5.3 MB", "MP3"))
    tree.insert(music_item, "end", text="Song2.wav", values=("12.7 MB", "WAV"))
    tree.insert(music_item, "end", text="Song3.flac", values=("18.9 MB", "FLAC"))
    
    # Scrollbar
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Control frame
    control_frame = tk.Frame(root)
    control_frame.pack(fill="x", padx=20, pady=10)
    
    def add_item():
        selected = tree.selection()
        if selected:
            parent = selected[0]
            name = f"New Item {len(tree.get_children(parent)) + 1}"
            tree.insert(parent, "end", text=name, values=("1.0 MB", "File"))
        else:
            name = f"New Item {len(tree.get_children()) + 1}"
            tree.insert("", "end", text=name, values=("1.0 MB", "File"))
    
    def remove_item():
        selected = tree.selection()
        if selected:
            tree.delete(selected[0])
        else:
            messagebox.showwarning("Warning", "Please select an item to remove")
    
    def clear_tree():
        for item in tree.get_children():
            tree.delete(item)
    
    tk.Button(control_frame, text="Add Item", command=add_item, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(control_frame, text="Remove Item", command=remove_item, bg="lightcoral").pack(side="left", padx=5)
    tk.Button(control_frame, text="Clear Tree", command=clear_tree, bg="lightyellow").pack(side="left", padx=5)
    
    root.mainloop()

# Custom widget
def custom_widget():
    """Create custom widget example"""
    root = tk.Tk()
    root.title("Custom Widget")
    root.geometry("500x400")
    
    # Custom widget class
    class CustomWidget(tk.Frame):
        def __init__(self, parent, **kwargs):
            super().__init__(parent, **kwargs)
            self.setup_widget()
        
        def setup_widget(self):
            # Title
            self.title_label = tk.Label(self, text="Custom Widget", font=("Arial", 14, "bold"))
            self.title_label.pack(pady=10)
            
            # Input frame
            input_frame = tk.Frame(self)
            input_frame.pack(pady=10)
            
            tk.Label(input_frame, text="Value:").pack(side="left")
            self.value_entry = tk.Entry(input_frame, width=10)
            self.value_entry.pack(side="left", padx=5)
            
            # Buttons
            button_frame = tk.Frame(self)
            button_frame.pack(pady=10)
            
            tk.Button(button_frame, text="Set", command=self.set_value, bg="lightblue").pack(side="left", padx=5)
            tk.Button(button_frame, text="Get", command=self.get_value, bg="lightgreen").pack(side="left", padx=5)
            tk.Button(button_frame, text="Clear", command=self.clear_value, bg="lightcoral").pack(side="left", padx=5)
            
            # Display
            self.display_label = tk.Label(self, text="No value set", font=("Arial", 12))
            self.display_label.pack(pady=10)
        
        def set_value(self):
            value = self.value_entry.get()
            if value:
                self.display_label.config(text=f"Value set to: {value}")
            else:
                messagebox.showwarning("Warning", "Please enter a value")
        
        def get_value(self):
            value = self.value_entry.get()
            if value:
                messagebox.showinfo("Value", f"Current value: {value}")
            else:
                messagebox.showwarning("Warning", "No value set")
        
        def clear_value(self):
            self.value_entry.delete(0, tk.END)
            self.display_label.config(text="No value set")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Create custom widget
    custom_widget = CustomWidget(main_frame, bg="lightgray", relief="raised", bd=2)
    custom_widget.pack(expand=True, fill="both")
    
    root.mainloop()

# Complete advanced widgets example
def complete_advanced_widgets_example():
    """Complete example showing all advanced widget features"""
    root = tk.Tk()
    root.title("Complete Advanced Widgets Example")
    root.geometry("800x700")
    
    # Header
    header = tk.Label(root, text="Complete Advanced Widgets Example", 
                     font=("Arial", 16, "bold"), bg="darkblue", fg="white")
    header.pack(fill="x", padx=10, pady=10)
    
    # Main content frame
    content_frame = tk.Frame(root, bg="lightgray")
    content_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Left column
    left_frame = tk.Frame(content_frame, bg="lightblue", relief="raised", bd=2)
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_frame, text="Progress & Scales", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
    
    # Progress bar
    progress = ttk.Progressbar(left_frame, mode="determinate", length=200)
    progress.pack(pady=10)
    progress["value"] = 50
    
    # Scale
    scale = tk.Scale(left_frame, from_=0, to=100, orient="horizontal", length=200)
    scale.pack(pady=10)
    scale.set(50)
    
    # Spinbox
    spinbox = tk.Spinbox(left_frame, from_=0, to=100, width=10)
    spinbox.pack(pady=10)
    spinbox.set(50)
    
    # Right column
    right_frame = tk.Frame(content_frame, bg="lightgreen", relief="raised", bd=2)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_frame, text="Lists & Trees", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=10)
    
    # Listbox
    listbox = tk.Listbox(right_frame, height=8)
    listbox.pack(pady=10, fill="x")
    
    for i in range(1, 11):
        listbox.insert(tk.END, f"Item {i}")
    
    # Treeview
    tree = ttk.Treeview(right_frame, columns=("value",), show="tree headings", height=8)
    tree.pack(pady=10, fill="x")
    
    tree.heading("#0", text="Name")
    tree.heading("value", text="Value")
    
    tree.column("#0", width=100)
    tree.column("value", width=100)
    
    for i in range(1, 6):
        tree.insert("", "end", text=f"Item {i}", values=(f"Value {i}",))
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Advanced Widgets Examples")
    print("=" * 40)
    print("1. Progress Bar")
    print("2. Scale Widget")
    print("3. Spinbox Widget")
    print("4. Notebook Widget")
    print("5. Treeview Widget")
    print("6. Custom Widget")
    print("7. Complete Example")
    print("=" * 40)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        progress_bar()
    elif choice == "2":
        scale_widget()
    elif choice == "3":
        spinbox_widget()
    elif choice == "4":
        notebook_widget()
    elif choice == "5":
        treeview_widget()
    elif choice == "6":
        custom_widget()
    elif choice == "7":
        complete_advanced_widgets_example()
    else:
        print("Invalid choice. Running progress bar...")
        progress_bar()

"""
Advanced Widget Types:
----------------------
- ProgressBar: Task progress indication
- Scale: Value selection slider
- Spinbox: Numeric input with arrows
- Notebook: Tabbed interface
- Treeview: Hierarchical data display
- Custom Widgets: User-defined components

Widget Properties:
------------------
- ProgressBar: mode, maximum, value
- Scale: from_, to, orient, resolution
- Spinbox: from_, to, values, increment
- Notebook: tabs, tab management
- Treeview: columns, headings, items

Best Practices:
--------------
- Use appropriate widgets for specific tasks
- Provide clear labels and instructions
- Consider accessibility and usability
- Test widgets with different data types
- Use consistent styling across your application
- Handle widget events appropriately

Extra Tips:
-----------
- Use ttk widgets for modern styling
- Use threading for long-running tasks
- Use event handling for widget interactions
- Use validation for user input
- Consider using third-party widgets
- Use custom widgets for specialized functionality
"""