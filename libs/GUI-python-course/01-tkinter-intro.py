"""
01-tkinter-intro.py

Complete beginner's guide to Tkinter GUI programming.

Overview:
---------
Tkinter is Python's built-in library for creating graphical user interfaces (GUIs). 
With Tkinter, you can build windows, dialogs, buttons, text boxes, and more—making 
your programs interactive and user-friendly.

What is Tkinter?
----------------
Tkinter is a wrapper around the Tcl/Tk GUI toolkit. It provides classes and functions 
to create windows and add widgets (GUI elements). Every Tkinter app starts by creating 
a main window (root).

How Tkinter Works:
------------------
1. Create a main window (root)
2. Add widgets (buttons, labels, etc.)
3. Configure widget properties
4. Start the event loop (mainloop)

Key Concepts:
--------------
- Root window: The main application window
- Widgets: GUI elements (buttons, labels, etc.)
- Event loop: Keeps the window responsive
- Geometry: Window size and position
- Title: Window title bar text

Examples:
---------
"""

import tkinter as tk
from tkinter import messagebox

# 1. Basic Tkinter window
def basic_window():
    """Create a basic Tkinter window"""
    # Create the main window
    root = tk.Tk()
    
    # Set window properties
    root.title("My First Tkinter App")
    root.geometry("400x300")
    
    # Add a label
    label = tk.Label(root, text="Hello, Tkinter!")
    label.pack()
    
    # Start the event loop
    root.mainloop()

# 2. Window with custom properties
def custom_window():
    """Create a window with custom properties"""
    root = tk.Tk()
    
    # Window properties
    root.title("Custom Window")
    root.geometry("500x400")
    root.resizable(True, True)  # Allow resizing
    root.configure(bg="lightblue")  # Background color
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Add content
    label = tk.Label(root, text="Custom Window", font=("Arial", 16), bg="lightblue")
    label.pack(pady=20)
    
    root.mainloop()

# 3. Window with multiple widgets
def multi_widget_window():
    """Create a window with multiple widgets"""
    root = tk.Tk()
    root.title("Multiple Widgets")
    root.geometry("400x300")
    
    # Title label
    title_label = tk.Label(root, text="Welcome to Tkinter!", font=("Arial", 18, "bold"))
    title_label.pack(pady=10)
    
    # Info label
    info_label = tk.Label(root, text="This window contains multiple widgets")
    info_label.pack(pady=5)
    
    # Button
    def button_click():
        messagebox.showinfo("Button Clicked", "Hello from the button!")
    
    button = tk.Button(root, text="Click Me!", command=button_click, bg="lightgreen")
    button.pack(pady=10)
    
    # Entry field
    entry = tk.Entry(root, width=30)
    entry.pack(pady=5)
    entry.insert(0, "Type something here...")
    
    # Text area
    text_area = tk.Text(root, height=5, width=40)
    text_area.pack(pady=10)
    text_area.insert("1.0", "This is a text area.\nYou can type multiple lines here.")
    
    root.mainloop()

# 4. Window with event handling
def event_handling_window():
    """Create a window with event handling"""
    root = tk.Tk()
    root.title("Event Handling")
    root.geometry("400x300")
    
    # Variables
    click_count = 0
    
    # Click counter
    def on_click():
        nonlocal click_count
        click_count += 1
        counter_label.config(text=f"Clicks: {click_count}")
    
    # Mouse enter/leave
    def on_enter(event):
        button.config(bg="lightblue")
    
    def on_leave(event):
        button.config(bg="lightgreen")
    
    # Create widgets
    counter_label = tk.Label(root, text="Clicks: 0", font=("Arial", 14))
    counter_label.pack(pady=20)
    
    button = tk.Button(root, text="Click Me!", command=on_click, bg="lightgreen")
    button.pack(pady=10)
    
    # Bind mouse events
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    
    # Keyboard events
    def on_key(event):
        if event.keysym == "Return":
            on_click()
    
    root.bind("<Key>", on_key)
    root.focus_set()  # Enable keyboard focus
    
    root.mainloop()

# 5. Professional window with menu
def professional_window():
    """Create a professional window with menu bar"""
    root = tk.Tk()
    root.title("Professional Tkinter App")
    root.geometry("600x400")
    root.configure(bg="white")
    
    # Menu bar
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
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Tkinter Professional App"))
    
    # Main content
    main_frame = tk.Frame(root, bg="white")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Title
    title_label = tk.Label(main_frame, text="Professional Tkinter Application", 
                          font=("Arial", 20, "bold"), bg="white")
    title_label.pack(pady=20)
    
    # Description
    desc_label = tk.Label(main_frame, 
                         text="This is a professional Tkinter application with menu bar,\n"
                              "proper layout, and event handling.",
                         font=("Arial", 12), bg="white")
    desc_label.pack(pady=10)
    
    # Buttons frame
    buttons_frame = tk.Frame(main_frame, bg="white")
    buttons_frame.pack(pady=20)
    
    # Action buttons
    btn1 = tk.Button(buttons_frame, text="Action 1", width=15, height=2)
    btn1.pack(side="left", padx=5)
    
    btn2 = tk.Button(buttons_frame, text="Action 2", width=15, height=2)
    btn2.pack(side="left", padx=5)
    
    btn3 = tk.Button(buttons_frame, text="Action 3", width=15, height=2)
    btn3.pack(side="left", padx=5)
    
    # Status bar
    status_bar = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status_bar.pack(side="bottom", fill="x")
    
    root.mainloop()

# 6. Window with custom styling
def styled_window():
    """Create a window with custom styling"""
    root = tk.Tk()
    root.title("Styled Window")
    root.geometry("500x400")
    root.configure(bg="#2c3e50")
    
    # Custom styles
    style = {
        "bg": "#2c3e50",
        "fg": "#ecf0f1",
        "font": ("Arial", 12)
    }
    
    button_style = {
        "bg": "#3498db",
        "fg": "white",
        "font": ("Arial", 12, "bold"),
        "relief": "raised",
        "bd": 2
    }
    
    # Header
    header = tk.Label(root, text="Styled Tkinter Application", 
                     font=("Arial", 18, "bold"), **style)
    header.pack(pady=20)
    
    # Content frame
    content_frame = tk.Frame(root, bg="#2c3e50")
    content_frame.pack(expand=True, fill="both", padx=20, pady=10)
    
    # Info text
    info_text = tk.Text(content_frame, height=8, width=50, 
                       bg="#34495e", fg="#ecf0f1", font=("Arial", 10))
    info_text.pack(pady=10)
    info_text.insert("1.0", "This is a styled Tkinter application with:\n\n"
                           "• Custom colors and fonts\n"
                           "• Professional appearance\n"
                           "• Modern design elements\n"
                           "• Responsive layout")
    
    # Buttons frame
    buttons_frame = tk.Frame(content_frame, bg="#2c3e50")
    buttons_frame.pack(pady=20)
    
    # Styled buttons
    btn1 = tk.Button(buttons_frame, text="Primary Action", **button_style)
    btn1.pack(side="left", padx=5)
    
    btn2 = tk.Button(buttons_frame, text="Secondary", 
                    bg="#e74c3c", fg="white", font=("Arial", 12, "bold"))
    btn2.pack(side="left", padx=5)
    
    btn3 = tk.Button(buttons_frame, text="Success", 
                    bg="#27ae60", fg="white", font=("Arial", 12, "bold"))
    btn3.pack(side="left", padx=5)
    
    root.mainloop()

# 7. Window with configuration
def configurable_window():
    """Create a window with configuration options"""
    root = tk.Tk()
    root.title("Configurable Window")
    root.geometry("400x300")
    
    # Configuration variables
    config = {
        "title": "My App",
        "width": 400,
        "height": 300,
        "bg_color": "white",
        "text_color": "black"
    }
    
    # Apply configuration
    root.title(config["title"])
    root.geometry(f"{config['width']}x{config['height']}")
    root.configure(bg=config["bg_color"])
    
    # Dynamic content
    def update_config():
        root.title(config["title"])
        root.geometry(f"{config['width']}x{config['height']}")
        root.configure(bg=config["bg_color"])
        label.config(text=config["title"], bg=config["bg_color"], fg=config["text_color"])
    
    # Main label
    label = tk.Label(root, text=config["title"], font=("Arial", 16))
    label.pack(pady=20)
    
    # Configuration buttons
    config_frame = tk.Frame(root, bg=config["bg_color"])
    config_frame.pack(pady=10)
    
    def change_title():
        config["title"] = "Updated Title"
        update_config()
    
    def change_size():
        config["width"] = 500
        config["height"] = 400
        update_config()
    
    def change_color():
        config["bg_color"] = "lightblue"
        config["text_color"] = "darkblue"
        update_config()
    
    # Buttons
    tk.Button(config_frame, text="Change Title", command=change_title).pack(side="left", padx=5)
    tk.Button(config_frame, text="Change Size", command=change_size).pack(side="left", padx=5)
    tk.Button(config_frame, text="Change Color", command=change_color).pack(side="left", padx=5)
    
    root.mainloop()

# 8. Window with error handling
def error_handling_window():
    """Create a window with error handling"""
    root = tk.Tk()
    root.title("Error Handling Window")
    root.geometry("400x300")
    
    def safe_operation():
        try:
            # Simulate an operation that might fail
            result = 10 / 0
            messagebox.showinfo("Success", f"Result: {result}")
        except ZeroDivisionError:
            messagebox.showerror("Error", "Cannot divide by zero!")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
    
    def safe_file_operation():
        try:
            with open("nonexistent.txt", "r") as file:
                content = file.read()
            messagebox.showinfo("Success", "File read successfully")
        except FileNotFoundError:
            messagebox.showerror("Error", "File not found!")
        except Exception as e:
            messagebox.showerror("Error", f"File error: {e}")
    
    # Main content
    title_label = tk.Label(root, text="Error Handling Demo", font=("Arial", 16, "bold"))
    title_label.pack(pady=20)
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    tk.Button(button_frame, text="Safe Operation", command=safe_operation, 
              bg="lightgreen").pack(pady=5)
    tk.Button(button_frame, text="File Operation", command=safe_file_operation, 
              bg="lightblue").pack(pady=5)
    
    # Info label
    info_label = tk.Label(root, text="Click buttons to test error handling", 
                         font=("Arial", 10))
    info_label.pack(pady=10)
    
    root.mainloop()

# 9. Window with layout management
def layout_window():
    """Create a window demonstrating layout management"""
    root = tk.Tk()
    root.title("Layout Management")
    root.geometry("600x500")
    
    # Pack layout example
    pack_frame = tk.LabelFrame(root, text="Pack Layout", font=("Arial", 12, "bold"))
    pack_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(pack_frame, text="Top", bg="red", fg="white").pack(fill="x", pady=2)
    tk.Label(pack_frame, text="Middle", bg="green", fg="white").pack(fill="x", pady=2)
    tk.Label(pack_frame, text="Bottom", bg="blue", fg="white").pack(fill="x", pady=2)
    
    # Grid layout example
    grid_frame = tk.LabelFrame(root, text="Grid Layout", font=("Arial", 12, "bold"))
    grid_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    # Grid buttons
    for i in range(3):
        for j in range(3):
            btn = tk.Button(grid_frame, text=f"{i},{j}", width=8, height=2)
            btn.grid(row=i, column=j, padx=2, pady=2)
    
    # Place layout example
    place_frame = tk.LabelFrame(root, text="Place Layout", font=("Arial", 12, "bold"))
    place_frame.pack(fill="x", padx=5, pady=5)
    
    tk.Label(place_frame, text="Absolute", bg="yellow").place(x=50, y=20, width=100, height=30)
    tk.Label(place_frame, text="Positioning", bg="orange").place(x=200, y=20, width=100, height=30)
    
    root.mainloop()

# 10. Complete application example
def complete_application():
    """Create a complete Tkinter application"""
    class TkinterApp:
        def __init__(self):
            self.root = tk.Tk()
            self.setup_window()
            self.create_widgets()
            self.setup_menu()
            self.setup_bindings()
        
        def setup_window(self):
            """Setup the main window"""
            self.root.title("Complete Tkinter Application")
            self.root.geometry("700x500")
            self.root.configure(bg="white")
            
            # Center window
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        def create_widgets(self):
            """Create all widgets"""
            # Header
            self.header = tk.Label(self.root, text="Complete Tkinter Application", 
                                  font=("Arial", 20, "bold"), bg="white")
            self.header.pack(pady=20)
            
            # Main content frame
            self.content_frame = tk.Frame(self.root, bg="white")
            self.content_frame.pack(expand=True, fill="both", padx=20, pady=10)
            
            # Input section
            input_frame = tk.LabelFrame(self.content_frame, text="Input", font=("Arial", 12, "bold"))
            input_frame.pack(fill="x", pady=10)
            
            tk.Label(input_frame, text="Name:").pack(side="left", padx=5)
            self.name_entry = tk.Entry(input_frame, width=30)
            self.name_entry.pack(side="left", padx=5)
            
            tk.Label(input_frame, text="Age:").pack(side="left", padx=5)
            self.age_entry = tk.Entry(input_frame, width=10)
            self.age_entry.pack(side="left", padx=5)
            
            # Buttons
            button_frame = tk.Frame(self.content_frame, bg="white")
            button_frame.pack(pady=10)
            
            tk.Button(button_frame, text="Add", command=self.add_person, 
                    bg="lightgreen").pack(side="left", padx=5)
            tk.Button(button_frame, text="Clear", command=self.clear_all, 
                    bg="lightcoral").pack(side="left", padx=5)
            tk.Button(button_frame, text="Save", command=self.save_data, 
                    bg="lightblue").pack(side="left", padx=5)
            
            # Listbox
            list_frame = tk.LabelFrame(self.content_frame, text="People", font=("Arial", 12, "bold"))
            list_frame.pack(fill="both", expand=True, pady=10)
            
            self.people_list = tk.Listbox(list_frame, height=10)
            self.people_list.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Status bar
            self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief="sunken", anchor="w")
            self.status_bar.pack(side="bottom", fill="x")
        
        def setup_menu(self):
            """Setup menu bar"""
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # File menu
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="New", command=self.new_file)
            file_menu.add_command(label="Open", command=self.open_file)
            file_menu.add_command(label="Save", command=self.save_data)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self.root.quit)
            
            # Edit menu
            edit_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Edit", menu=edit_menu)
            edit_menu.add_command(label="Clear", command=self.clear_all)
            edit_menu.add_command(label="Delete", command=self.delete_selected)
            
            # Help menu
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="About", command=self.show_about)
        
        def setup_bindings(self):
            """Setup keyboard bindings"""
            self.root.bind("<Control-n>", lambda e: self.new_file())
            self.root.bind("<Control-s>", lambda e: self.save_data())
            self.root.bind("<Delete>", lambda e: self.delete_selected())
        
        def add_person(self):
            """Add a person to the list"""
            name = self.name_entry.get().strip()
            age = self.age_entry.get().strip()
            
            if name and age:
                try:
                    age_int = int(age)
                    person = f"{name} ({age_int} years old)"
                    self.people_list.insert(tk.END, person)
                    self.name_entry.delete(0, tk.END)
                    self.age_entry.delete(0, tk.END)
                    self.status_bar.config(text="Person added successfully")
                except ValueError:
                    messagebox.showerror("Error", "Age must be a number")
            else:
                messagebox.showwarning("Warning", "Please enter both name and age")
        
        def clear_all(self):
            """Clear all data"""
            self.people_list.delete(0, tk.END)
            self.name_entry.delete(0, tk.END)
            self.age_entry.delete(0, tk.END)
            self.status_bar.config(text="All data cleared")
        
        def delete_selected(self):
            """Delete selected item"""
            selection = self.people_list.curselection()
            if selection:
                self.people_list.delete(selection[0])
                self.status_bar.config(text="Item deleted")
            else:
                messagebox.showwarning("Warning", "Please select an item to delete")
        
        def save_data(self):
            """Save data to file"""
            try:
                with open("people.txt", "w") as file:
                    for i in range(self.people_list.size()):
                        file.write(self.people_list.get(i) + "\n")
                self.status_bar.config(text="Data saved successfully")
                messagebox.showinfo("Success", "Data saved to people.txt")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
        
        def new_file(self):
            """Create new file"""
            self.clear_all()
            self.status_bar.config(text="New file created")
        
        def open_file(self):
            """Open existing file"""
            try:
                with open("people.txt", "r") as file:
                    self.clear_all()
                    for line in file:
                        self.people_list.insert(tk.END, line.strip())
                self.status_bar.config(text="File opened successfully")
            except FileNotFoundError:
                messagebox.showwarning("Warning", "No saved file found")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {e}")
        
        def show_about(self):
            """Show about dialog"""
            messagebox.showinfo("About", "Complete Tkinter Application\nVersion 1.0\n\nA demonstration of Tkinter capabilities")
        
        def run(self):
            """Run the application"""
            self.root.mainloop()
    
    # Create and run the application
    app = TkinterApp()
    app.run()

# Main execution
if __name__ == "__main__":
    print("Tkinter Introduction Examples")
    print("=" * 40)
    print("1. Basic Window")
    print("2. Custom Window")
    print("3. Multi-Widget Window")
    print("4. Event Handling Window")
    print("5. Professional Window")
    print("6. Styled Window")
    print("7. Configurable Window")
    print("8. Error Handling Window")
    print("9. Layout Window")
    print("10. Complete Application")
    print("=" * 40)
    
    choice = input("Enter your choice (1-10): ")
    
    if choice == "1":
        basic_window()
    elif choice == "2":
        custom_window()
    elif choice == "3":
        multi_widget_window()
    elif choice == "4":
        event_handling_window()
    elif choice == "5":
        professional_window()
    elif choice == "6":
        styled_window()
    elif choice == "7":
        configurable_window()
    elif choice == "8":
        error_handling_window()
    elif choice == "9":
        layout_window()
    elif choice == "10":
        complete_application()
    else:
        print("Invalid choice. Running basic window...")
        basic_window()

"""
Tkinter Fundamentals:
--------------------
- Root window: The main application window
- Widgets: GUI elements (buttons, labels, etc.)
- Event loop: Keeps the window responsive
- Geometry: Window size and position
- Title: Window title bar text

Essential Methods:
------------------
- tk.Tk(): Create main window
- root.title(): Set window title
- root.geometry(): Set window size
- root.mainloop(): Start event loop
- widget.pack(): Add widget to window
- widget.grid(): Add widget with grid layout
- widget.place(): Add widget with absolute positioning

Common Widgets:
---------------
- tk.Label: Display text or images
- tk.Button: Clickable buttons
- tk.Entry: Single-line text input
- tk.Text: Multi-line text input
- tk.Frame: Container for other widgets
- tk.Listbox: List of selectable items
- tk.Menu: Menu bars and context menus

Layout Managers:
----------------
- pack(): Simple vertical/horizontal layout
- grid(): Table-like layout with rows and columns
- place(): Absolute positioning

Event Handling:
---------------
- command: Function to call when widget is activated
- bind(): Bind events to functions
- mainloop(): Start the event loop

Best Practices:
---------------
- Always call mainloop() to start the application
- Use meaningful variable names for widgets
- Organize code with functions and classes
- Handle errors gracefully
- Use appropriate layout managers
- Test your application thoroughly

"""