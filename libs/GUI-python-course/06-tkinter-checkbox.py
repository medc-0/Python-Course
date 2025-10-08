"""
06-tkinter-checkbox.py

Tkinter Checkbox - Boolean input widgets for user choices

Overview:
---------
Checkboxes allow users to select multiple options from a list. They're 
perfect for settings, preferences, and any situation where users need 
to make multiple selections.

Key Features:
- Multiple selection support
- Custom styling and colors
- Disabled/enabled states
- Event handling for state changes
- Grouped checkboxes for related options

Common Use Cases:
- Settings and preferences
- Form checkboxes and agreements
- Feature toggles and options
- Multi-select lists
- Terms and conditions acceptance

Tips:
- Use clear, descriptive labels
- Group related checkboxes together
- Provide visual feedback for state changes
- Use consistent styling across your app
- Consider keyboard accessibility
"""

import tkinter as tk
from tkinter import messagebox

# Basic checkbox examples
def basic_checkboxes():
    """Create basic checkboxes"""
    root = tk.Tk()
    root.title("Basic Checkboxes")
    root.geometry("400x300")
    
    # Create checkboxes
    var1 = tk.BooleanVar()
    var2 = tk.BooleanVar()
    var3 = tk.BooleanVar()
    
    cb1 = tk.Checkbutton(root, text="Option 1", variable=var1)
    cb1.pack(pady=5, anchor="w")
    
    cb2 = tk.Checkbutton(root, text="Option 2", variable=var2)
    cb2.pack(pady=5, anchor="w")
    
    cb3 = tk.Checkbutton(root, text="Option 3", variable=var3)
    cb3.pack(pady=5, anchor="w")
    
    # Submit button
    def submit_choices():
        choices = []
        if var1.get():
            choices.append("Option 1")
        if var2.get():
            choices.append("Option 2")
        if var3.get():
            choices.append("Option 3")
        
        if choices:
            messagebox.showinfo("Selected Options", f"You selected: {', '.join(choices)}")
        else:
            messagebox.showwarning("No Selection", "Please select at least one option")
    
    tk.Button(root, text="Submit", command=submit_choices, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Checkbox with event handling
def event_checkboxes():
    """Create checkboxes with event handling"""
    root = tk.Tk()
    root.title("Event Handling Checkboxes")
    root.geometry("400x350")
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 12))
    status_label.pack(pady=10)
    
    # Checkbox variables
    var1 = tk.BooleanVar()
    var2 = tk.BooleanVar()
    var3 = tk.BooleanVar()
    
    # Event handlers
    def on_checkbox_change():
        selected = []
        if var1.get():
            selected.append("Option 1")
        if var2.get():
            selected.append("Option 2")
        if var3.get():
            selected.append("Option 3")
        
        if selected:
            status_label.config(text=f"Selected: {', '.join(selected)}", fg="green")
        else:
            status_label.config(text="No options selected", fg="red")
    
    # Create checkboxes with event binding
    cb1 = tk.Checkbutton(root, text="Option 1", variable=var1, command=on_checkbox_change)
    cb1.pack(pady=5, anchor="w")
    
    cb2 = tk.Checkbutton(root, text="Option 2", variable=var2, command=on_checkbox_change)
    cb2.pack(pady=5, anchor="w")
    
    cb3 = tk.Checkbutton(root, text="Option 3", variable=var3, command=on_checkbox_change)
    cb3.pack(pady=5, anchor="w")
    
    # Select all button
    def select_all():
        var1.set(True)
        var2.set(True)
        var3.set(True)
        on_checkbox_change()
    
    # Deselect all button
    def deselect_all():
        var1.set(False)
        var2.set(False)
        var3.set(False)
        on_checkbox_change()
    
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Select All", command=select_all, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Deselect All", command=deselect_all, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Styled checkboxes
def styled_checkboxes():
    """Create checkboxes with custom styling"""
    root = tk.Tk()
    root.title("Styled Checkboxes")
    root.geometry("400x350")
    
    # Create styled checkboxes
    var1 = tk.BooleanVar()
    var2 = tk.BooleanVar()
    var3 = tk.BooleanVar()
    
    # Styled checkbox 1
    cb1 = tk.Checkbutton(root, text="Bold Option", variable=var1,
                         font=("Arial", 12, "bold"), fg="blue", bg="lightblue")
    cb1.pack(pady=5, anchor="w", padx=10)
    
    # Styled checkbox 2
    cb2 = tk.Checkbutton(root, text="Italic Option", variable=var2,
                         font=("Arial", 12, "italic"), fg="green", bg="lightgreen")
    cb2.pack(pady=5, anchor="w", padx=10)
    
    # Styled checkbox 3
    cb3 = tk.Checkbutton(root, text="Underlined Option", variable=var3,
                         font=("Arial", 12, "underline"), fg="red", bg="lightcoral")
    cb3.pack(pady=5, anchor="w", padx=10)
    
    # Submit button
    def submit_choices():
        choices = []
        if var1.get():
            choices.append("Bold Option")
        if var2.get():
            choices.append("Italic Option")
        if var3.get():
            choices.append("Underlined Option")
        
        if choices:
            messagebox.showinfo("Selected Options", f"You selected: {', '.join(choices)}")
        else:
            messagebox.showwarning("No Selection", "Please select at least one option")
    
    tk.Button(root, text="Submit", command=submit_choices, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Grouped checkboxes
def grouped_checkboxes():
    """Create grouped checkboxes for related options"""
    root = tk.Tk()
    root.title("Grouped Checkboxes")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Personal Information group
    personal_frame = tk.LabelFrame(main_frame, text="Personal Information", font=("Arial", 12, "bold"))
    personal_frame.pack(fill="x", pady=10)
    
    personal_vars = []
    personal_options = ["Name", "Email", "Phone", "Address"]
    
    for option in personal_options:
        var = tk.BooleanVar()
        personal_vars.append(var)
        cb = tk.Checkbutton(personal_frame, text=option, variable=var)
        cb.pack(anchor="w", padx=10, pady=2)
    
    # Preferences group
    preferences_frame = tk.LabelFrame(main_frame, text="Preferences", font=("Arial", 12, "bold"))
    preferences_frame.pack(fill="x", pady=10)
    
    preferences_vars = []
    preferences_options = ["Newsletter", "SMS Notifications", "Email Notifications", "Push Notifications"]
    
    for option in preferences_options:
        var = tk.BooleanVar()
        preferences_vars.append(var)
        cb = tk.Checkbutton(preferences_frame, text=option, variable=var)
        cb.pack(anchor="w", padx=10, pady=2)
    
    # Submit button
    def submit_choices():
        personal_selected = []
        preferences_selected = []
        
        for i, var in enumerate(personal_vars):
            if var.get():
                personal_selected.append(personal_options[i])
        
        for i, var in enumerate(preferences_vars):
            if var.get():
                preferences_selected.append(preferences_options[i])
        
        result = f"Personal: {', '.join(personal_selected) if personal_selected else 'None'}\n"
        result += f"Preferences: {', '.join(preferences_selected) if preferences_selected else 'None'}"
        
        messagebox.showinfo("Selected Options", result)
    
    tk.Button(main_frame, text="Submit", command=submit_choices, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Checkbox with validation
def validation_checkboxes():
    """Create checkboxes with validation"""
    root = tk.Tk()
    root.title("Validation Checkboxes")
    root.geometry("400x350")
    
    # Create checkboxes
    var1 = tk.BooleanVar()
    var2 = tk.BooleanVar()
    var3 = tk.BooleanVar()
    
    cb1 = tk.Checkbutton(root, text="I agree to the terms and conditions", variable=var1)
    cb1.pack(pady=5, anchor="w")
    
    cb2 = tk.Checkbutton(root, text="I agree to receive marketing emails", variable=var2)
    cb2.pack(pady=5, anchor="w")
    
    cb3 = tk.Checkbutton(root, text="I confirm that I am over 18 years old", variable=var3)
    cb3.pack(pady=5, anchor="w")
    
    # Validation function
    def validate_choices():
        errors = []
        
        if not var1.get():
            errors.append("You must agree to the terms and conditions")
        
        if not var3.get():
            errors.append("You must confirm that you are over 18 years old")
        
        if errors:
            error_message = "Please fix the following errors:\n\n" + "\n".join(errors)
            messagebox.showerror("Validation Error", error_message)
        else:
            messagebox.showinfo("Success", "All validations passed!")
    
    # Submit button
    tk.Button(root, text="Submit", command=validate_choices, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Checkbox with dynamic content
def dynamic_checkboxes():
    """Create checkboxes with dynamic content"""
    root = tk.Tk()
    root.title("Dynamic Checkboxes")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Options list
    options = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
    checkbox_vars = []
    checkboxes = []
    
    # Create checkboxes dynamically
    for i, option in enumerate(options):
        var = tk.BooleanVar()
        checkbox_vars.append(var)
        
        cb = tk.Checkbutton(main_frame, text=option, variable=var)
        cb.pack(anchor="w", padx=10, pady=2)
        checkboxes.append(cb)
    
    # Add new option button
    def add_option():
        new_option = f"Option {len(options) + 1}"
        options.append(new_option)
        
        var = tk.BooleanVar()
        checkbox_vars.append(var)
        
        cb = tk.Checkbutton(main_frame, text=new_option, variable=var)
        cb.pack(anchor="w", padx=10, pady=2)
        checkboxes.append(cb)
    
    # Remove last option button
    def remove_option():
        if len(options) > 1:
            options.pop()
            checkbox_vars.pop()
            checkboxes.pop().destroy()
    
    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Add Option", command=add_option, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Remove Option", command=remove_option, bg="lightcoral").pack(side="left", padx=5)
    
    # Submit button
    def submit_choices():
        selected = []
        for i, var in enumerate(checkbox_vars):
            if var.get():
                selected.append(options[i])
        
        if selected:
            messagebox.showinfo("Selected Options", f"You selected: {', '.join(selected)}")
        else:
            messagebox.showwarning("No Selection", "Please select at least one option")
    
    tk.Button(main_frame, text="Submit", command=submit_choices, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Complete checkbox example
def complete_checkbox_example():
    """Complete example showing all checkbox features"""
    root = tk.Tk()
    root.title("Complete Checkbox Example")
    root.geometry("600x500")
    
    # Header
    header = tk.Label(root, text="Complete Checkbox Example", 
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
    
    # Left column checkboxes
    tk.Label(left_frame, text="Basic Options:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    var1 = tk.BooleanVar()
    var2 = tk.BooleanVar()
    var3 = tk.BooleanVar()
    
    tk.Checkbutton(left_frame, text="Option 1", variable=var1).pack(anchor="w", pady=2)
    tk.Checkbutton(left_frame, text="Option 2", variable=var2).pack(anchor="w", pady=2)
    tk.Checkbutton(left_frame, text="Option 3", variable=var3).pack(anchor="w", pady=2)
    
    # Right column checkboxes
    tk.Label(right_frame, text="Styled Options:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    var4 = tk.BooleanVar()
    var5 = tk.BooleanVar()
    var6 = tk.BooleanVar()
    
    tk.Checkbutton(right_frame, text="Bold Option", variable=var4, 
                  font=("Arial", 10, "bold"), fg="blue").pack(anchor="w", pady=2)
    tk.Checkbutton(right_frame, text="Italic Option", variable=var5, 
                  font=("Arial", 10, "italic"), fg="green").pack(anchor="w", pady=2)
    tk.Checkbutton(right_frame, text="Underlined Option", variable=var6, 
                  font=("Arial", 10, "underline"), fg="red").pack(anchor="w", pady=2)
    
    # Submit button
    def submit_choices():
        selected = []
        if var1.get():
            selected.append("Option 1")
        if var2.get():
            selected.append("Option 2")
        if var3.get():
            selected.append("Option 3")
        if var4.get():
            selected.append("Bold Option")
        if var5.get():
            selected.append("Italic Option")
        if var6.get():
            selected.append("Underlined Option")
        
        if selected:
            messagebox.showinfo("Selected Options", f"You selected: {', '.join(selected)}")
        else:
            messagebox.showwarning("No Selection", "Please select at least one option")
    
    tk.Button(content_frame, text="Submit", command=submit_choices, bg="lightblue").pack(pady=20)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Checkbox Examples")
    print("=" * 30)
    print("1. Basic Checkboxes")
    print("2. Event Handling Checkboxes")
    print("3. Styled Checkboxes")
    print("4. Grouped Checkboxes")
    print("5. Validation Checkboxes")
    print("6. Dynamic Checkboxes")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_checkboxes()
    elif choice == "2":
        event_checkboxes()
    elif choice == "3":
        styled_checkboxes()
    elif choice == "4":
        grouped_checkboxes()
    elif choice == "5":
        validation_checkboxes()
    elif choice == "6":
        dynamic_checkboxes()
    elif choice == "7":
        complete_checkbox_example()
    else:
        print("Invalid choice. Running basic checkboxes...")
        basic_checkboxes()

"""
Checkbox Properties:
--------------------
- text: Checkbox label text
- variable: BooleanVar to store state
- command: Function to call when state changes
- font: Font family, size, and style
- fg: Foreground (text) color
- bg: Background color
- state: "normal", "disabled", or "active"
- relief: Border style
- bd: Border width

Checkbox States:
----------------
- "normal": Default state, clickable
- "disabled": Grayed out, not clickable
- "active": Currently being pressed

Common Use Cases:
-----------------
- Settings and preferences
- Form checkboxes and agreements
- Feature toggles and options
- Multi-select lists
- Terms and conditions acceptance

Best Practices:
--------------
- Use clear, descriptive labels
- Group related checkboxes together
- Provide visual feedback for state changes
- Use consistent styling throughout your app
- Consider keyboard accessibility
- Test checkbox functionality thoroughly

Extra Tips:
-----------
- Use BooleanVar for two-way data binding
- Use command parameter for real-time updates
- Use state parameter to disable checkboxes
- Use relief and borderwidth for custom borders
- Consider using ttk.Checkbutton for modern styling
- Use compound for image and text positioning
"""