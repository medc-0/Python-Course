"""
10-tkinter-messagebox.py

Tkinter Messagebox - Dialog boxes for user interaction

Overview:
---------
Messageboxes are dialog boxes that display messages, warnings, errors, 
and questions to users. They're essential for user feedback and 
interaction in GUI applications.

Key Features:
- Various message types (info, warning, error, question)
- Custom titles and messages
- User response handling
- Modal dialog behavior
- Standard button layouts

Common Use Cases:
- Confirmation dialogs
- Error and warning messages
- Information display
- User input prompts
- Application notifications

Tips:
- Use appropriate message types for different situations
- Keep messages clear and concise
- Provide meaningful titles
- Handle user responses appropriately
- Consider accessibility for screen readers
"""

import tkinter as tk
from tkinter import messagebox

# Basic messagebox examples
def basic_messageboxes():
    """Create basic messageboxes"""
    root = tk.Tk()
    root.title("Basic Messageboxes")
    root.geometry("400x300")
    
    # Info messagebox
    def show_info():
        messagebox.showinfo("Information", "This is an information message.")
    
    # Warning messagebox
    def show_warning():
        messagebox.showwarning("Warning", "This is a warning message.")
    
    # Error messagebox
    def show_error():
        messagebox.showerror("Error", "This is an error message.")
    
    # Question messagebox
    def show_question():
        result = messagebox.askquestion("Question", "Do you want to continue?")
        if result == "yes":
            messagebox.showinfo("Response", "You clicked Yes!")
        else:
            messagebox.showinfo("Response", "You clicked No!")
    
    # Yes/No messagebox
    def show_yesno():
        result = messagebox.askyesno("Yes/No", "Do you agree to the terms?")
        if result:
            messagebox.showinfo("Response", "You agreed!")
        else:
            messagebox.showinfo("Response", "You disagreed!")
    
    # OK/Cancel messagebox
    def show_okcancel():
        result = messagebox.askokcancel("OK/Cancel", "Do you want to proceed?")
        if result:
            messagebox.showinfo("Response", "You clicked OK!")
        else:
            messagebox.showinfo("Response", "You clicked Cancel!")
    
    # Retry/Cancel messagebox
    def show_retrycancel():
        result = messagebox.askretrycancel("Retry/Cancel", "Operation failed. Retry?")
        if result:
            messagebox.showinfo("Response", "You clicked Retry!")
        else:
            messagebox.showinfo("Response", "You clicked Cancel!")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Show Info", command=show_info, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Show Warning", command=show_warning, bg="lightyellow").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Show Error", command=show_error, bg="lightcoral").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Show Question", command=show_question, bg="lightgreen").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Show Yes/No", command=show_yesno, bg="lightgreen").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Show OK/Cancel", command=show_okcancel, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Show Retry/Cancel", command=show_retrycancel, bg="lightcoral").pack(pady=5, fill="x")
    
    root.mainloop()

# Messagebox with custom content
def custom_messageboxes():
    """Create messageboxes with custom content"""
    root = tk.Tk()
    root.title("Custom Messageboxes")
    root.geometry("400x300")
    
    # Custom info messagebox
    def show_custom_info():
        messagebox.showinfo("Custom Info", "This is a custom information message with detailed content.")
    
    # Custom warning messagebox
    def show_custom_warning():
        messagebox.showwarning("Custom Warning", "This is a custom warning message with important details.")
    
    # Custom error messagebox
    def show_custom_error():
        messagebox.showerror("Custom Error", "This is a custom error message with error details.")
    
    # Custom question messagebox
    def show_custom_question():
        result = messagebox.askquestion("Custom Question", "This is a custom question message. Do you want to continue?")
        if result == "yes":
            messagebox.showinfo("Response", "You clicked Yes!")
        else:
            messagebox.showinfo("Response", "You clicked No!")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Custom Info", command=show_custom_info, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Custom Warning", command=show_custom_warning, bg="lightyellow").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Custom Error", command=show_custom_error, bg="lightcoral").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Custom Question", command=show_custom_question, bg="lightgreen").pack(pady=5, fill="x")
    
    root.mainloop()

# Messagebox with validation
def validation_messageboxes():
    """Create messageboxes with validation"""
    root = tk.Tk()
    root.title("Validation Messageboxes")
    root.geometry("400x300")
    
    # Validation function
    def validate_input():
        # Simulate input validation
        input_value = "test"  # Simulate user input
        
        if not input_value:
            messagebox.showerror("Validation Error", "Input is required!")
        elif len(input_value) < 3:
            messagebox.showwarning("Validation Warning", "Input must be at least 3 characters long!")
        else:
            messagebox.showinfo("Validation Success", "Input is valid!")
    
    # Confirmation function
    def confirm_action():
        result = messagebox.askyesno("Confirmation", "Are you sure you want to perform this action?")
        if result:
            messagebox.showinfo("Action Confirmed", "Action has been performed!")
        else:
            messagebox.showinfo("Action Cancelled", "Action has been cancelled!")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Validate Input", command=validate_input, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Confirm Action", command=confirm_action, bg="lightgreen").pack(pady=5, fill="x")
    
    root.mainloop()

# Messagebox with error handling
def error_handling_messageboxes():
    """Create messageboxes with error handling"""
    root = tk.Tk()
    root.title("Error Handling Messageboxes")
    root.geometry("400x300")
    
    # Error handling function
    def handle_error():
        try:
            # Simulate an error
            result = 10 / 0
        except ZeroDivisionError:
            messagebox.showerror("Error", "Cannot divide by zero!")
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {e}")
    
    # File error handling
    def handle_file_error():
        try:
            # Simulate file operation
            with open("nonexistent.txt", "r") as file:
                content = file.read()
        except FileNotFoundError:
            messagebox.showerror("File Error", "File not found!")
        except Exception as e:
            messagebox.showerror("File Error", f"File operation failed: {e}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Handle Division Error", command=handle_error, bg="lightcoral").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Handle File Error", command=handle_file_error, bg="lightcoral").pack(pady=5, fill="x")
    
    root.mainloop()

# Messagebox with user input
def input_messageboxes():
    """Create messageboxes with user input"""
    root = tk.Tk()
    root.title("Input Messageboxes")
    root.geometry("400x300")
    
    # Simple input dialog
    def show_simple_input():
        from tkinter import simpledialog
        name = simpledialog.askstring("Input", "Enter your name:")
        if name:
            messagebox.showinfo("Greeting", f"Hello, {name}!")
        else:
            messagebox.showwarning("No Input", "No name entered!")
    
    # Integer input dialog
    def show_integer_input():
        from tkinter import simpledialog
        age = simpledialog.askinteger("Input", "Enter your age:", minvalue=0, maxvalue=120)
        if age is not None:
            messagebox.showinfo("Age", f"You are {age} years old!")
        else:
            messagebox.showwarning("No Input", "No age entered!")
    
    # Float input dialog
    def show_float_input():
        from tkinter import simpledialog
        price = simpledialog.askfloat("Input", "Enter price:", minvalue=0.0)
        if price is not None:
            messagebox.showinfo("Price", f"Price: ${price:.2f}")
        else:
            messagebox.showwarning("No Input", "No price entered!")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Simple Input", command=show_simple_input, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Integer Input", command=show_integer_input, bg="lightgreen").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Float Input", command=show_float_input, bg="lightyellow").pack(pady=5, fill="x")
    
    root.mainloop()

# Messagebox with file operations
def file_operation_messageboxes():
    """Create messageboxes with file operations"""
    root = tk.Tk()
    root.title("File Operation Messageboxes")
    root.geometry("400x300")
    
    # Save file confirmation
    def save_file():
        result = messagebox.askyesno("Save File", "Do you want to save the current file?")
        if result:
            messagebox.showinfo("File Saved", "File has been saved successfully!")
        else:
            messagebox.showinfo("Save Cancelled", "File save has been cancelled!")
    
    # Delete file confirmation
    def delete_file():
        result = messagebox.askyesno("Delete File", "Are you sure you want to delete this file?")
        if result:
            messagebox.showinfo("File Deleted", "File has been deleted!")
        else:
            messagebox.showinfo("Delete Cancelled", "File deletion has been cancelled!")
    
    # Open file confirmation
    def open_file():
        result = messagebox.askyesno("Open File", "Do you want to open a new file?")
        if result:
            messagebox.showinfo("File Opened", "File has been opened successfully!")
        else:
            messagebox.showinfo("Open Cancelled", "File opening has been cancelled!")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Save File", command=save_file, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Delete File", command=delete_file, bg="lightcoral").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Open File", command=open_file, bg="lightgreen").pack(pady=5, fill="x")
    
    root.mainloop()

# Complete messagebox example
def complete_messagebox_example():
    """Complete example showing all messagebox features"""
    root = tk.Tk()
    root.title("Complete Messagebox Example")
    root.geometry("600x500")
    
    # Header
    header = tk.Label(root, text="Complete Messagebox Example", 
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
    tk.Label(left_frame, text="Basic Messageboxes:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    tk.Button(left_frame, text="Info", command=lambda: messagebox.showinfo("Info", "This is an info message"), 
              bg="lightblue").pack(pady=2, fill="x")
    tk.Button(left_frame, text="Warning", command=lambda: messagebox.showwarning("Warning", "This is a warning message"), 
              bg="lightyellow").pack(pady=2, fill="x")
    tk.Button(left_frame, text="Error", command=lambda: messagebox.showerror("Error", "This is an error message"), 
              bg="lightcoral").pack(pady=2, fill="x")
    
    # Right column buttons
    tk.Label(right_frame, text="Interactive Messageboxes:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    tk.Button(right_frame, text="Question", command=lambda: messagebox.askquestion("Question", "Do you want to continue?"), 
              bg="lightgreen").pack(pady=2, fill="x")
    tk.Button(right_frame, text="Yes/No", command=lambda: messagebox.askyesno("Yes/No", "Do you agree?"), 
              bg="lightgreen").pack(pady=2, fill="x")
    tk.Button(right_frame, text="OK/Cancel", command=lambda: messagebox.askokcancel("OK/Cancel", "Do you want to proceed?"), 
              bg="lightblue").pack(pady=2, fill="x")
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Messagebox Examples")
    print("=" * 30)
    print("1. Basic Messageboxes")
    print("2. Custom Messageboxes")
    print("3. Validation Messageboxes")
    print("4. Error Handling Messageboxes")
    print("5. Input Messageboxes")
    print("6. File Operation Messageboxes")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_messageboxes()
    elif choice == "2":
        custom_messageboxes()
    elif choice == "3":
        validation_messageboxes()
    elif choice == "4":
        error_handling_messageboxes()
    elif choice == "5":
        input_messageboxes()
    elif choice == "6":
        file_operation_messageboxes()
    elif choice == "7":
        complete_messagebox_example()
    else:
        print("Invalid choice. Running basic messageboxes...")
        basic_messageboxes()

"""
Messagebox Types:
-----------------
- showinfo(): Information message
- showwarning(): Warning message
- showerror(): Error message
- askquestion(): Yes/No question
- askyesno(): Yes/No question
- askokcancel(): OK/Cancel question
- askretrycancel(): Retry/Cancel question

Messagebox Properties:
----------------------
- title: Dialog box title
- message: Dialog box message
- icon: Icon type (info, warning, error, question)
- type: Button type (ok, okcancel, yesno, yesnocancel, retrycancel, abortretryignore)

Common Use Cases:
-----------------
- Confirmation dialogs
- Error and warning messages
- Information display
- User input prompts
- Application notifications

Best Practices:
--------------
- Use appropriate message types for different situations
- Keep messages clear and concise
- Provide meaningful titles
- Handle user responses appropriately
- Consider accessibility for screen readers
- Test messagebox functionality thoroughly

Extra Tips:
-----------
- Use messagebox for user feedback
- Use simpledialog for user input
- Use filedialog for file operations
- Use colorchooser for color selection
- Use fontchooser for font selection
- Consider using custom dialogs for complex interactions
"""