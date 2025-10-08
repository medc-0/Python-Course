"""
17-tkinter-menu.py

Tkinter Menu - Menu bars and context menus

Overview:
---------
Menus provide a standard way to organize application functionality. 
They include menu bars, context menus, and popup menus that help 
users navigate and access features in your application.

Key Features:
- Menu bars with cascading menus
- Context menus and popup menus
- Keyboard shortcuts and accelerators
- Menu separators and submenus
- Dynamic menu content

Common Use Cases:
- Application menu bars
- Context menus for right-click actions
- Popup menus for user interactions
- Navigation and feature access
- Settings and preferences

Tips:
- Use clear, descriptive menu labels
- Group related menu items together
- Provide keyboard shortcuts for common actions
- Use separators to organize menu sections
- Consider accessibility and usability
"""

import tkinter as tk
from tkinter import messagebox, filedialog

# Basic menu bar
def basic_menu_bar():
    """Create basic menu bar"""
    root = tk.Tk()
    root.title("Basic Menu Bar")
    root.geometry("500x400")
    
    # Create menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="New", command=lambda: messagebox.showinfo("File", "New file"))
    file_menu.add_command(label="Open", command=lambda: messagebox.showinfo("File", "Open file"))
    file_menu.add_command(label="Save", command=lambda: messagebox.showinfo("File", "Save file"))
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    # Edit menu
    edit_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Edit", menu=edit_menu)
    edit_menu.add_command(label="Cut", command=lambda: messagebox.showinfo("Edit", "Cut"))
    edit_menu.add_command(label="Copy", command=lambda: messagebox.showinfo("Edit", "Copy"))
    edit_menu.add_command(label="Paste", command=lambda: messagebox.showinfo("Edit", "Paste"))
    
    # View menu
    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Zoom In", command=lambda: messagebox.showinfo("View", "Zoom In"))
    view_menu.add_command(label="Zoom Out", command=lambda: messagebox.showinfo("View", "Zoom Out"))
    view_menu.add_command(label="Reset Zoom", command=lambda: messagebox.showinfo("View", "Reset Zoom"))
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Menu Bar Example"))
    
    # Main content
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Label(main_frame, text="Basic Menu Bar Example", 
             font=("Arial", 16, "bold"), bg="lightgray").pack(expand=True)
    
    root.mainloop()

# Menu with keyboard shortcuts
def menu_with_shortcuts():
    """Create menu with keyboard shortcuts"""
    root = tk.Tk()
    root.title("Menu with Keyboard Shortcuts")
    root.geometry("500x400")
    
    # Create menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="New", command=lambda: messagebox.showinfo("File", "New file (Ctrl+N)"), 
                         accelerator="Ctrl+N")
    file_menu.add_command(label="Open", command=lambda: messagebox.showinfo("File", "Open file (Ctrl+O)"), 
                         accelerator="Ctrl+O")
    file_menu.add_command(label="Save", command=lambda: messagebox.showinfo("File", "Save file (Ctrl+S)"), 
                         accelerator="Ctrl+S")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit, accelerator="Ctrl+Q")
    
    # Edit menu
    edit_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Edit", menu=edit_menu)
    edit_menu.add_command(label="Cut", command=lambda: messagebox.showinfo("Edit", "Cut (Ctrl+X)"), 
                         accelerator="Ctrl+X")
    edit_menu.add_command(label="Copy", command=lambda: messagebox.showinfo("Edit", "Copy (Ctrl+C)"), 
                         accelerator="Ctrl+C")
    edit_menu.add_command(label="Paste", command=lambda: messagebox.showinfo("Edit", "Paste (Ctrl+V)"), 
                         accelerator="Ctrl+V")
    
    # Keyboard bindings
    root.bind("<Control-n>", lambda e: messagebox.showinfo("File", "New file (Ctrl+N)"))
    root.bind("<Control-o>", lambda e: messagebox.showinfo("File", "Open file (Ctrl+O)"))
    root.bind("<Control-s>", lambda e: messagebox.showinfo("File", "Save file (Ctrl+S)"))
    root.bind("<Control-q>", lambda e: root.quit())
    root.bind("<Control-x>", lambda e: messagebox.showinfo("Edit", "Cut (Ctrl+X)"))
    root.bind("<Control-c>", lambda e: messagebox.showinfo("Edit", "Copy (Ctrl+C)"))
    root.bind("<Control-v>", lambda e: messagebox.showinfo("Edit", "Paste (Ctrl+V)"))
    
    # Main content
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Label(main_frame, text="Menu with Keyboard Shortcuts", 
             font=("Arial", 16, "bold"), bg="lightgray").pack(expand=True)
    
    root.mainloop()

# Context menu
def context_menu():
    """Create context menu"""
    root = tk.Tk()
    root.title("Context Menu")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Create text widget
    text_widget = tk.Text(main_frame, height=15, width=50, wrap="word")
    text_widget.pack(expand=True, fill="both")
    
    # Add some content
    text_content = """This is a text widget with a context menu.

Right-click anywhere in the text area to see the context menu.
The context menu provides quick access to common text operations.

Try right-clicking to see the menu options!"""
    
    text_widget.insert("1.0", text_content)
    
    # Create context menu
    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_command(label="Cut", command=lambda: messagebox.showinfo("Context", "Cut selected text"))
    context_menu.add_command(label="Copy", command=lambda: messagebox.showinfo("Context", "Copy selected text"))
    context_menu.add_command(label="Paste", command=lambda: messagebox.showinfo("Context", "Paste text"))
    context_menu.add_separator()
    context_menu.add_command(label="Select All", command=lambda: messagebox.showinfo("Context", "Select all text"))
    context_menu.add_command(label="Clear", command=lambda: messagebox.showinfo("Context", "Clear text"))
    context_menu.add_separator()
    context_menu.add_command(label="Find", command=lambda: messagebox.showinfo("Context", "Find text"))
    context_menu.add_command(label="Replace", command=lambda: messagebox.showinfo("Context", "Replace text"))
    
    # Bind context menu to text widget
    def show_context_menu(event):
        context_menu.post(event.x_root, event.y_root)
    
    text_widget.bind("<Button-3>", show_context_menu)  # Right-click
    
    root.mainloop()

# Popup menu
def popup_menu():
    """Create popup menu"""
    root = tk.Tk()
    root.title("Popup Menu")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Create canvas
    canvas = tk.Canvas(main_frame, bg="white", relief="sunken", bd=2)
    canvas.pack(expand=True, fill="both")
    
    # Add some content to canvas
    canvas.create_rectangle(50, 50, 150, 100, fill="lightblue", outline="blue", width=2)
    canvas.create_oval(200, 50, 300, 100, fill="lightgreen", outline="green", width=2)
    canvas.create_text(175, 150, text="Right-click anywhere to see popup menu", 
                      font=("Arial", 12), fill="purple")
    
    # Create popup menu
    popup_menu = tk.Menu(root, tearoff=0)
    popup_menu.add_command(label="Add Rectangle", command=lambda: messagebox.showinfo("Popup", "Add Rectangle"))
    popup_menu.add_command(label="Add Circle", command=lambda: messagebox.showinfo("Popup", "Add Circle"))
    popup_menu.add_command(label="Add Text", command=lambda: messagebox.showinfo("Popup", "Add Text"))
    popup_menu.add_separator()
    popup_menu.add_command(label="Clear All", command=lambda: messagebox.showinfo("Popup", "Clear All"))
    popup_menu.add_command(label="Properties", command=lambda: messagebox.showinfo("Popup", "Properties"))
    
    # Bind popup menu to canvas
    def show_popup_menu(event):
        popup_menu.post(event.x_root, event.y_root)
    
    canvas.bind("<Button-3>", show_popup_menu)  # Right-click
    
    root.mainloop()

# Dynamic menu
def dynamic_menu():
    """Create dynamic menu"""
    root = tk.Tk()
    root.title("Dynamic Menu")
    root.geometry("500x400")
    
    # Create menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="New", command=lambda: messagebox.showinfo("File", "New file"))
    file_menu.add_command(label="Open", command=lambda: messagebox.showinfo("File", "Open file"))
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    # Dynamic menu
    dynamic_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Dynamic", menu=dynamic_menu)
    
    # Menu items list
    menu_items = ["Item 1", "Item 2", "Item 3"]
    
    def update_menu():
        # Clear existing items
        dynamic_menu.delete(0, tk.END)
        
        # Add current items
        for item in menu_items:
            dynamic_menu.add_command(label=item, command=lambda i=item: messagebox.showinfo("Dynamic", f"Selected: {i}"))
        
        dynamic_menu.add_separator()
        dynamic_menu.add_command(label="Add Item", command=add_item)
        dynamic_menu.add_command(label="Remove Item", command=remove_item)
    
    def add_item():
        new_item = f"Item {len(menu_items) + 1}"
        menu_items.append(new_item)
        update_menu()
        messagebox.showinfo("Dynamic", f"Added: {new_item}")
    
    def remove_item():
        if menu_items:
            removed_item = menu_items.pop()
            update_menu()
            messagebox.showinfo("Dynamic", f"Removed: {removed_item}")
        else:
            messagebox.showwarning("Dynamic", "No items to remove")
    
    # Initialize menu
    update_menu()
    
    # Main content
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Label(main_frame, text="Dynamic Menu Example", 
             font=("Arial", 16, "bold"), bg="lightgray").pack(expand=True)
    
    root.mainloop()

# Menu with submenus
def menu_with_submenus():
    """Create menu with submenus"""
    root = tk.Tk()
    root.title("Menu with Submenus")
    root.geometry("500x400")
    
    # Create menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="New", command=lambda: messagebox.showinfo("File", "New file"))
    file_menu.add_command(label="Open", command=lambda: messagebox.showinfo("File", "Open file"))
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    # Edit menu
    edit_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Edit", menu=edit_menu)
    edit_menu.add_command(label="Cut", command=lambda: messagebox.showinfo("Edit", "Cut"))
    edit_menu.add_command(label="Copy", command=lambda: messagebox.showinfo("Edit", "Copy"))
    edit_menu.add_command(label="Paste", command=lambda: messagebox.showinfo("Edit", "Paste"))
    
    # View menu
    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Zoom In", command=lambda: messagebox.showinfo("View", "Zoom In"))
    view_menu.add_command(label="Zoom Out", command=lambda: messagebox.showinfo("View", "Zoom Out"))
    view_menu.add_separator()
    
    # View submenu
    view_submenu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Toolbars", menu=view_submenu)
    view_submenu.add_command(label="Standard", command=lambda: messagebox.showinfo("View", "Standard Toolbar"))
    view_submenu.add_command(label="Formatting", command=lambda: messagebox.showinfo("View", "Formatting Toolbar"))
    view_submenu.add_command(label="Drawing", command=lambda: messagebox.showinfo("View", "Drawing Toolbar"))
    
    # Tools menu
    tools_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Tools", menu=tools_menu)
    tools_menu.add_command(label="Options", command=lambda: messagebox.showinfo("Tools", "Options"))
    tools_menu.add_command(label="Customize", command=lambda: messagebox.showinfo("Tools", "Customize"))
    tools_menu.add_separator()
    
    # Tools submenu
    tools_submenu = tk.Menu(tools_menu, tearoff=0)
    tools_menu.add_cascade(label="External Tools", menu=tools_submenu)
    tools_submenu.add_command(label="Calculator", command=lambda: messagebox.showinfo("Tools", "Calculator"))
    tools_submenu.add_command(label="Notepad", command=lambda: messagebox.showinfo("Tools", "Notepad"))
    tools_submenu.add_command(label="Command Prompt", command=lambda: messagebox.showinfo("Tools", "Command Prompt"))
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="Contents", command=lambda: messagebox.showinfo("Help", "Contents"))
    help_menu.add_command(label="Index", command=lambda: messagebox.showinfo("Help", "Index"))
    help_menu.add_separator()
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Menu with Submenus Example"))
    
    # Main content
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Label(main_frame, text="Menu with Submenus Example", 
             font=("Arial", 16, "bold"), bg="lightgray").pack(expand=True)
    
    root.mainloop()

# Complete menu example
def complete_menu_example():
    """Complete example showing all menu features"""
    root = tk.Tk()
    root.title("Complete Menu Example")
    root.geometry("700x600")
    
    # Create menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="New", command=lambda: messagebox.showinfo("File", "New file"), accelerator="Ctrl+N")
    file_menu.add_command(label="Open", command=lambda: messagebox.showinfo("File", "Open file"), accelerator="Ctrl+O")
    file_menu.add_command(label="Save", command=lambda: messagebox.showinfo("File", "Save file"), accelerator="Ctrl+S")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit, accelerator="Ctrl+Q")
    
    # Edit menu
    edit_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Edit", menu=edit_menu)
    edit_menu.add_command(label="Cut", command=lambda: messagebox.showinfo("Edit", "Cut"), accelerator="Ctrl+X")
    edit_menu.add_command(label="Copy", command=lambda: messagebox.showinfo("Edit", "Copy"), accelerator="Ctrl+C")
    edit_menu.add_command(label="Paste", command=lambda: messagebox.showinfo("Edit", "Paste"), accelerator="Ctrl+V")
    edit_menu.add_separator()
    edit_menu.add_command(label="Select All", command=lambda: messagebox.showinfo("Edit", "Select All"), accelerator="Ctrl+A")
    
    # View menu
    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Zoom In", command=lambda: messagebox.showinfo("View", "Zoom In"))
    view_menu.add_command(label="Zoom Out", command=lambda: messagebox.showinfo("View", "Zoom Out"))
    view_menu.add_command(label="Reset Zoom", command=lambda: messagebox.showinfo("View", "Reset Zoom"))
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="Contents", command=lambda: messagebox.showinfo("Help", "Contents"))
    help_menu.add_command(label="Index", command=lambda: messagebox.showinfo("Help", "Index"))
    help_menu.add_separator()
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Complete Menu Example"))
    
    # Keyboard bindings
    root.bind("<Control-n>", lambda e: messagebox.showinfo("File", "New file (Ctrl+N)"))
    root.bind("<Control-o>", lambda e: messagebox.showinfo("File", "Open file (Ctrl+O)"))
    root.bind("<Control-s>", lambda e: messagebox.showinfo("File", "Save file (Ctrl+S)"))
    root.bind("<Control-q>", lambda e: root.quit())
    root.bind("<Control-x>", lambda e: messagebox.showinfo("Edit", "Cut (Ctrl+X)"))
    root.bind("<Control-c>", lambda e: messagebox.showinfo("Edit", "Copy (Ctrl+C)"))
    root.bind("<Control-v>", lambda e: messagebox.showinfo("Edit", "Paste (Ctrl+V)"))
    root.bind("<Control-a>", lambda e: messagebox.showinfo("Edit", "Select All (Ctrl+A)"))
    
    # Main content
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Left column
    left_frame = tk.Frame(main_frame, bg="lightblue", relief="raised", bd=2)
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_frame, text="Menu Features", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
    
    tk.Label(left_frame, text="• Menu bars with cascading menus", bg="lightblue").pack(anchor="w", padx=10)
    tk.Label(left_frame, text="• Keyboard shortcuts and accelerators", bg="lightblue").pack(anchor="w", padx=10)
    tk.Label(left_frame, text="• Context menus and popup menus", bg="lightblue").pack(anchor="w", padx=10)
    tk.Label(left_frame, text="• Dynamic menu content", bg="lightblue").pack(anchor="w", padx=10)
    tk.Label(left_frame, text="• Menu separators and submenus", bg="lightblue").pack(anchor="w", padx=10)
    
    # Right column
    right_frame = tk.Frame(main_frame, bg="lightgreen", relief="raised", bd=2)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_frame, text="Usage Tips", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=10)
    
    tk.Label(right_frame, text="• Use clear, descriptive menu labels", bg="lightgreen").pack(anchor="w", padx=10)
    tk.Label(right_frame, text="• Group related menu items together", bg="lightgreen").pack(anchor="w", padx=10)
    tk.Label(right_frame, text="• Provide keyboard shortcuts", bg="lightgreen").pack(anchor="w", padx=10)
    tk.Label(right_frame, text="• Use separators to organize sections", bg="lightgreen").pack(anchor="w", padx=10)
    tk.Label(right_frame, text="• Consider accessibility and usability", bg="lightgreen").pack(anchor="w", padx=10)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Menu Examples")
    print("=" * 30)
    print("1. Basic Menu Bar")
    print("2. Menu with Shortcuts")
    print("3. Context Menu")
    print("4. Popup Menu")
    print("5. Dynamic Menu")
    print("6. Menu with Submenus")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_menu_bar()
    elif choice == "2":
        menu_with_shortcuts()
    elif choice == "3":
        context_menu()
    elif choice == "4":
        popup_menu()
    elif choice == "5":
        dynamic_menu()
    elif choice == "6":
        menu_with_submenus()
    elif choice == "7":
        complete_menu_example()
    else:
        print("Invalid choice. Running basic menu bar...")
        basic_menu_bar()

"""
Menu Types:
-----------
- Menu: Main menu bar
- Context Menu: Right-click menu
- Popup Menu: Popup menu
- Submenu: Nested menu

Menu Properties:
----------------
- tearoff: Allow menu to be torn off
- accelerator: Keyboard shortcut display
- command: Function to call when selected
- label: Menu item text
- state: Menu item state (normal, disabled)

Menu Methods:
-------------
- add_command(): Add menu item
- add_separator(): Add separator
- add_cascade(): Add submenu
- delete(): Remove menu item
- insert(): Insert menu item

Best Practices:
--------------
- Use clear, descriptive menu labels
- Group related menu items together
- Provide keyboard shortcuts for common actions
- Use separators to organize menu sections
- Consider accessibility and usability
- Test menu functionality thoroughly

Extra Tips:
-----------
- Use accelerator parameter for keyboard shortcuts
- Use bind() for keyboard event handling
- Use post() for popup menus
- Use tearoff=0 to disable tear-off
- Consider using ttk.Menu for modern styling
- Use state parameter to disable menu items
"""