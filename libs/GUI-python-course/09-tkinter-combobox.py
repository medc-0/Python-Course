"""
09-tkinter-combobox.py

Tkinter Combobox - Dropdown selection widgets

Overview:
---------
Comboboxes combine the functionality of an Entry widget with a dropdown 
list. They're perfect for providing a list of predefined options while 
allowing users to type custom values.

Key Features:
- Dropdown list of predefined options
- Custom text input capability
- Search and filter functionality
- Custom styling and colors
- Event handling for selection changes

Common Use Cases:
- Country/state selection
- Category selection
- Search with suggestions
- Form dropdowns
- Settings and preferences

Tips:
- Use clear, descriptive option labels
- Provide search functionality for long lists
- Handle both selection and custom input
- Use appropriate default values
- Consider keyboard navigation
"""

import tkinter as tk
from tkinter import messagebox, ttk

# Basic combobox examples
def basic_combobox():
    """Create basic combobox"""
    root = tk.Tk()
    root.title("Basic Combobox")
    root.geometry("400x300")
    
    # Create combobox
    combo = ttk.Combobox(root, width=30)
    combo.pack(pady=20)
    
    # Set options
    options = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
    combo['values'] = options
    combo.set("Select an option")  # Set default text
    
    # Submit button
    def submit_selection():
        selected = combo.get()
        if selected and selected != "Select an option":
            messagebox.showinfo("Selected Option", f"You selected: {selected}")
        else:
            messagebox.showwarning("No Selection", "Please select an option")
    
    tk.Button(root, text="Submit", command=submit_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Combobox with event handling
def event_combobox():
    """Create combobox with event handling"""
    root = tk.Tk()
    root.title("Event Handling Combobox")
    root.geometry("400x350")
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 12))
    status_label.pack(pady=10)
    
    # Create combobox
    combo = ttk.Combobox(root, width=30)
    combo.pack(pady=10)
    
    # Set options
    options = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
    combo['values'] = options
    combo.set("Select an option")
    
    # Event handler
    def on_selection_change(event):
        selected = combo.get()
        if selected and selected != "Select an option":
            status_label.config(text=f"Selected: {selected}", fg="green")
        else:
            status_label.config(text="No selection", fg="red")
    
    # Bind event
    combo.bind("<<ComboboxSelected>>", on_selection_change)
    
    # Reset button
    def reset_selection():
        combo.set("Select an option")
        status_label.config(text="Status: Ready", fg="black")
    
    tk.Button(root, text="Reset", command=reset_selection, bg="lightcoral").pack(pady=10)
    
    root.mainloop()

# Styled combobox
def styled_combobox():
    """Create combobox with custom styling"""
    root = tk.Tk()
    root.title("Styled Combobox")
    root.geometry("400x350")
    
    # Create styled combobox
    combo = ttk.Combobox(root, width=30, font=("Arial", 12))
    combo.pack(pady=20)
    
    # Set options
    options = ["Styled Option 1", "Styled Option 2", "Styled Option 3", "Styled Option 4", "Styled Option 5"]
    combo['values'] = options
    combo.set("Select a styled option")
    
    # Submit button
    def submit_selection():
        selected = combo.get()
        if selected and selected != "Select a styled option":
            messagebox.showinfo("Selected Option", f"You selected: {selected}")
        else:
            messagebox.showwarning("No Selection", "Please select an option")
    
    tk.Button(root, text="Submit", command=submit_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Combobox with search functionality
def searchable_combobox():
    """Create combobox with search functionality"""
    root = tk.Tk()
    root.title("Searchable Combobox")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Search frame
    search_frame = tk.Frame(main_frame)
    search_frame.pack(fill="x", pady=(0, 10))
    
    tk.Label(search_frame, text="Search:").pack(side="left")
    search_entry = tk.Entry(search_frame, width=30)
    search_entry.pack(side="left", padx=5)
    
    # Create combobox
    combo = ttk.Combobox(main_frame, width=40)
    combo.pack(pady=10)
    
    # Set options
    options = [f"Option {i}" for i in range(1, 101)]  # 100 options
    combo['values'] = options
    combo.set("Select an option")
    
    # Search function
    def search_options():
        search_term = search_entry.get().lower()
        if search_term:
            filtered_options = [option for option in options if search_term in option.lower()]
            combo['values'] = filtered_options
        else:
            combo['values'] = options
    
    # Bind search to entry
    search_entry.bind("<KeyRelease>", lambda e: search_options())
    
    # Submit button
    def submit_selection():
        selected = combo.get()
        if selected and selected != "Select an option":
            messagebox.showinfo("Selected Option", f"You selected: {selected}")
        else:
            messagebox.showwarning("No Selection", "Please select an option")
    
    tk.Button(main_frame, text="Submit", command=submit_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Combobox with validation
def validation_combobox():
    """Create combobox with validation"""
    root = tk.Tk()
    root.title("Validation Combobox")
    root.geometry("400x350")
    
    # Create combobox
    combo = ttk.Combobox(root, width=30)
    combo.pack(pady=20)
    
    # Set options
    options = ["Valid Option 1", "Valid Option 2", "Valid Option 3", "Valid Option 4", "Valid Option 5"]
    combo['values'] = options
    combo.set("Select a valid option")
    
    # Validation function
    def validate_selection():
        selected = combo.get()
        if selected and selected != "Select a valid option":
            if selected in options:
                messagebox.showinfo("Success", f"Valid selection: {selected}")
            else:
                messagebox.showerror("Error", f"Invalid selection: {selected}\nPlease select from the dropdown list.")
        else:
            messagebox.showwarning("No Selection", "Please select an option")
    
    # Submit button
    tk.Button(root, text="Validate", command=validate_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Combobox with dynamic content
def dynamic_combobox():
    """Create combobox with dynamic content"""
    root = tk.Tk()
    root.title("Dynamic Combobox")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Create combobox
    combo = ttk.Combobox(main_frame, width=40)
    combo.pack(pady=10)
    
    # Initial options
    options = ["Option 1", "Option 2", "Option 3"]
    combo['values'] = options
    combo.set("Select an option")
    
    # Add new option button
    def add_option():
        new_option = f"Option {len(options) + 1}"
        options.append(new_option)
        combo['values'] = options
        combo.set(new_option)
    
    # Remove last option button
    def remove_option():
        if len(options) > 1:
            options.pop()
            combo['values'] = options
            combo.set("Select an option")
    
    # Clear all options button
    def clear_options():
        options.clear()
        combo['values'] = []
        combo.set("No options available")
    
    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Add Option", command=add_option, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Remove Option", command=remove_option, bg="lightcoral").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear All", command=clear_options, bg="lightyellow").pack(side="left", padx=5)
    
    # Submit button
    def submit_selection():
        selected = combo.get()
        if selected and selected != "Select an option" and selected != "No options available":
            messagebox.showinfo("Selected Option", f"You selected: {selected}")
        else:
            messagebox.showwarning("No Selection", "Please select an option")
    
    tk.Button(main_frame, text="Submit", command=submit_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Complete combobox example
def complete_combobox_example():
    """Complete example showing all combobox features"""
    root = tk.Tk()
    root.title("Complete Combobox Example")
    root.geometry("600x500")
    
    # Header
    header = tk.Label(root, text="Complete Combobox Example", 
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
    
    # Left column combobox
    tk.Label(left_frame, text="Basic Combobox:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    combo1 = ttk.Combobox(left_frame, width=25)
    combo1.pack(pady=5)
    
    options1 = ["Basic Option 1", "Basic Option 2", "Basic Option 3", "Basic Option 4"]
    combo1['values'] = options1
    combo1.set("Select basic option")
    
    # Right column combobox
    tk.Label(right_frame, text="Styled Combobox:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    combo2 = ttk.Combobox(right_frame, width=25)
    combo2.pack(pady=5)
    
    options2 = ["Styled Option 1", "Styled Option 2", "Styled Option 3", "Styled Option 4"]
    combo2['values'] = options2
    combo2.set("Select styled option")
    
    # Submit button
    def submit_selection():
        selected1 = combo1.get()
        selected2 = combo2.get()
        
        result = "Selected Options:\n"
        if selected1 and selected1 != "Select basic option":
            result += f"Basic: {selected1}\n"
        if selected2 and selected2 != "Select styled option":
            result += f"Styled: {selected2}"
        
        if not (selected1 and selected1 != "Select basic option") and not (selected2 and selected2 != "Select styled option"):
            result = "No options selected"
        
        messagebox.showinfo("Selected Options", result)
    
    tk.Button(content_frame, text="Submit", command=submit_selection, bg="lightblue").pack(pady=20)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Combobox Examples")
    print("=" * 30)
    print("1. Basic Combobox")
    print("2. Event Handling Combobox")
    print("3. Styled Combobox")
    print("4. Searchable Combobox")
    print("5. Validation Combobox")
    print("6. Dynamic Combobox")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_combobox()
    elif choice == "2":
        event_combobox()
    elif choice == "3":
        styled_combobox()
    elif choice == "4":
        searchable_combobox()
    elif choice == "5":
        validation_combobox()
    elif choice == "6":
        dynamic_combobox()
    elif choice == "7":
        complete_combobox_example()
    else:
        print("Invalid choice. Running basic combobox...")
        basic_combobox()

"""
Combobox Properties:
--------------------
- values: List of options to display
- width: Width in characters
- font: Font family, size, and style
- state: "normal", "disabled", or "readonly"
- textvariable: Variable to store current value
- exportselection: Whether to export selection to clipboard

Combobox States:
----------------
- "normal": Default state, editable
- "disabled": Grayed out, not editable
- "readonly": Dropdown only, not editable

Common Events:
---------------
- <<ComboboxSelected>>: Fired when selection changes
- <KeyRelease>: Fired when key is released
- <FocusIn>: Fired when widget gains focus
- <FocusOut>: Fired when widget loses focus

Best Practices:
--------------
- Use clear, descriptive option labels
- Provide search functionality for long lists
- Handle both selection and custom input
- Use appropriate default values
- Consider keyboard navigation
- Test combobox functionality thoroughly

Extra Tips:
-----------
- Use textvariable for two-way data binding
- Use bind() for custom event handling
- Use state parameter to control editability
- Use exportselection for clipboard integration
- Consider using ttk.Combobox for modern styling
- Use values parameter for dynamic option updates
"""