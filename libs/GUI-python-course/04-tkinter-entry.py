"""
04-tkinter-entry.py

Tkinter Entry - Single-line text input fields

Overview:
---------
Entry widgets are used for single-line text input. They're perfect for 
forms, search boxes, and any place where users need to enter text.

Key Features:
- Single-line text input
- Text validation and formatting
- Password fields with hidden text
- Placeholder text and hints
- Input restrictions and limits

Common Use Cases:
- Username and password fields
- Search boxes and filters
- Form inputs (name, email, etc.)
- Numeric input with validation
- URL and path inputs

Tips:
- Use placeholder text to guide users
- Validate input before processing
- Provide clear error messages
- Use appropriate input types (text, password, etc.)
- Consider input length limits
"""

import tkinter as tk
from tkinter import messagebox

# Basic entry examples
def basic_entries():
    """Create basic entry fields"""
    root = tk.Tk()
    root.title("Tkinter Entry Fields")
    root.geometry("400x300")
    
    # Simple entry
    tk.Label(root, text="Enter your name:").pack(pady=5)
    name_entry = tk.Entry(root, width=30)
    name_entry.pack(pady=5)
    
    # Entry with placeholder
    tk.Label(root, text="Enter your email:").pack(pady=5)
    email_entry = tk.Entry(root, width=30)
    email_entry.pack(pady=5)
    email_entry.insert(0, "example@email.com")
    
    # Password entry
    tk.Label(root, text="Enter password:").pack(pady=5)
    password_entry = tk.Entry(root, width=30, show="*")
    password_entry.pack(pady=5)
    
    # Submit button
    def submit_form():
        name = name_entry.get()
        email = email_entry.get()
        password = password_entry.get()
        
        if name and email and password:
            messagebox.showinfo("Success", f"Hello {name}!\nEmail: {email}")
        else:
            messagebox.showwarning("Warning", "Please fill all fields")
    
    tk.Button(root, text="Submit", command=submit_form, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Entry with validation
def validation_entries():
    """Create entry fields with validation"""
    root = tk.Tk()
    root.title("Entry Validation")
    root.geometry("400x350")
    
    # Name validation
    def validate_name(event):
        name = name_entry.get()
        if len(name) < 2:
            name_label.config(text="Name must be at least 2 characters", fg="red")
        else:
            name_label.config(text="Name is valid", fg="green")
    
    tk.Label(root, text="Enter your name:").pack(pady=5)
    name_entry = tk.Entry(root, width=30)
    name_entry.pack(pady=5)
    name_entry.bind("<KeyRelease>", validate_name)
    
    name_label = tk.Label(root, text="")
    name_label.pack(pady=2)
    
    # Email validation
    def validate_email(event):
        email = email_entry.get()
        if "@" in email and "." in email:
            email_label.config(text="Email is valid", fg="green")
        else:
            email_label.config(text="Enter a valid email", fg="red")
    
    tk.Label(root, text="Enter your email:").pack(pady=5)
    email_entry = tk.Entry(root, width=30)
    email_entry.pack(pady=5)
    email_entry.bind("<KeyRelease>", validate_email)
    
    email_label = tk.Label(root, text="")
    email_label.pack(pady=2)
    
    # Age validation
    def validate_age(event):
        age = age_entry.get()
        if age.isdigit() and 0 < int(age) < 120:
            age_label.config(text="Age is valid", fg="green")
        else:
            age_label.config(text="Enter a valid age (1-119)", fg="red")
    
    tk.Label(root, text="Enter your age:").pack(pady=5)
    age_entry = tk.Entry(root, width=30)
    age_entry.pack(pady=5)
    age_entry.bind("<KeyRelease>", validate_age)
    
    age_label = tk.Label(root, text="")
    age_label.pack(pady=2)
    
    root.mainloop()

# Entry with different styles
def styled_entries():
    """Create entry fields with different styles"""
    root = tk.Tk()
    root.title("Styled Entry Fields")
    root.geometry("400x350")
    
    # Default style
    tk.Label(root, text="Default Style:").pack(pady=5)
    default_entry = tk.Entry(root, width=30)
    default_entry.pack(pady=5)
    
    # Custom font and colors
    tk.Label(root, text="Custom Style:").pack(pady=5)
    custom_entry = tk.Entry(root, width=30, font=("Arial", 12), 
                           fg="blue", bg="lightyellow")
    custom_entry.pack(pady=5)
    
    # Disabled entry
    tk.Label(root, text="Disabled Entry:").pack(pady=5)
    disabled_entry = tk.Entry(root, width=30, state="disabled")
    disabled_entry.pack(pady=5)
    
    # Read-only entry
    tk.Label(root, text="Read-only Entry:").pack(pady=5)
    readonly_entry = tk.Entry(root, width=30, state="readonly")
    readonly_entry.pack(pady=5)
    readonly_entry.config(state="normal")
    readonly_entry.insert(0, "This is read-only text")
    readonly_entry.config(state="readonly")
    
    # Entry with border
    tk.Label(root, text="Bordered Entry:").pack(pady=5)
    bordered_entry = tk.Entry(root, width=30, relief="sunken", bd=3)
    bordered_entry.pack(pady=5)
    
    root.mainloop()

# Entry with special characters
def special_entries():
    """Create entry fields with special characters"""
    root = tk.Tk()
    root.title("Special Entry Fields")
    root.geometry("400x300")
    
    # Password entry
    tk.Label(root, text="Password:").pack(pady=5)
    password_entry = tk.Entry(root, width=30, show="*")
    password_entry.pack(pady=5)
    
    # Hidden text entry
    tk.Label(root, text="Hidden Text:").pack(pady=5)
    hidden_entry = tk.Entry(root, width=30, show="•")
    hidden_entry.pack(pady=5)
    
    # Numeric entry
    tk.Label(root, text="Numeric Only:").pack(pady=5)
    numeric_entry = tk.Entry(root, width=30)
    numeric_entry.pack(pady=5)
    
    def validate_numeric(event):
        value = numeric_entry.get()
        if not value.isdigit() and value != "":
            numeric_entry.delete(0, tk.END)
            numeric_entry.insert(0, value[:-1])
    
    numeric_entry.bind("<KeyRelease>", validate_numeric)
    
    # Email entry
    tk.Label(root, text="Email:").pack(pady=5)
    email_entry = tk.Entry(root, width=30)
    email_entry.pack(pady=5)
    
    root.mainloop()

# Entry with events
def event_entries():
    """Create entry fields with event handling"""
    root = tk.Tk()
    root.title("Entry Events")
    root.geometry("400x350")
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 12))
    status_label.pack(pady=10)
    
    # Entry with focus events
    def on_focus_in(event):
        status_label.config(text="Status: Entry focused", fg="blue")
    
    def on_focus_out(event):
        status_label.config(text="Status: Entry unfocused", fg="gray")
    
    tk.Label(root, text="Enter text:").pack(pady=5)
    entry = tk.Entry(root, width=30)
    entry.pack(pady=5)
    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)
    
    # Entry with key events
    def on_key_press(event):
        status_label.config(text=f"Status: Key pressed: {event.keysym}", fg="green")
    
    def on_key_release(event):
        status_label.config(text=f"Status: Key released: {event.keysym}", fg="orange")
    
    tk.Label(root, text="Press keys:").pack(pady=5)
    key_entry = tk.Entry(root, width=30)
    key_entry.pack(pady=5)
    key_entry.bind("<KeyPress>", on_key_press)
    key_entry.bind("<KeyRelease>", on_key_release)
    
    root.mainloop()

# Entry with formatting
def formatting_entries():
    """Create entry fields with text formatting"""
    root = tk.Tk()
    root.title("Entry Formatting")
    root.geometry("400x350")
    
    # Phone number formatting
    def format_phone(event):
        phone = phone_entry.get()
        # Remove non-digits
        digits = ''.join(filter(str.isdigit, phone))
        if len(digits) >= 10:
            formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            phone_entry.delete(0, tk.END)
            phone_entry.insert(0, formatted)
    
    tk.Label(root, text="Phone Number:").pack(pady=5)
    phone_entry = tk.Entry(root, width=30)
    phone_entry.pack(pady=5)
    phone_entry.bind("<KeyRelease>", format_phone)
    
    # Currency formatting
    def format_currency(event):
        value = currency_entry.get()
        # Remove non-digits and decimal
        digits = ''.join(filter(lambda x: x.isdigit() or x == '.', value))
        if digits:
            try:
                formatted = f"${float(digits):.2f}"
                currency_entry.delete(0, tk.END)
                currency_entry.insert(0, formatted)
            except ValueError:
                pass
    
    tk.Label(root, text="Currency:").pack(pady=5)
    currency_entry = tk.Entry(root, width=30)
    currency_entry.pack(pady=5)
    currency_entry.bind("<KeyRelease>", format_currency)
    
    # Date formatting
    def format_date(event):
        date = date_entry.get()
        # Remove non-digits
        digits = ''.join(filter(str.isdigit, date))
        if len(digits) >= 8:
            formatted = f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"
            date_entry.delete(0, tk.END)
            date_entry.insert(0, formatted)
    
    tk.Label(root, text="Date (MM/DD/YYYY):").pack(pady=5)
    date_entry = tk.Entry(root, width=30)
    date_entry.pack(pady=5)
    date_entry.bind("<KeyRelease>", format_date)
    
    root.mainloop()

# Complete entry example
def complete_entry_example():
    """Complete example showing all entry features"""
    root = tk.Tk()
    root.title("Complete Entry Example")
    root.geometry("500x400")
    
    # Header
    header = tk.Label(root, text="Complete Entry Example", 
                     font=("Arial", 16, "bold"))
    header.pack(pady=10)
    
    # Main content frame
    content_frame = tk.Frame(root)
    content_frame.pack(expand=True, fill="both", padx=20, pady=10)
    
    # Form fields
    fields = [
        ("Name:", "text"),
        ("Email:", "email"),
        ("Phone:", "phone"),
        ("Password:", "password")
    ]
    
    entries = {}
    
    for label_text, field_type in fields:
        frame = tk.Frame(content_frame)
        frame.pack(fill="x", pady=5)
        
        label = tk.Label(frame, text=label_text, width=10, anchor="w")
        label.pack(side="left")
        
        if field_type == "password":
            entry = tk.Entry(frame, width=30, show="*")
        else:
            entry = tk.Entry(frame, width=30)
        
        entry.pack(side="left", padx=5)
        entries[field_type] = entry
    
    # Buttons
    button_frame = tk.Frame(content_frame)
    button_frame.pack(pady=20)
    
    def submit_form():
        data = {field_type: entry.get() for field_type, entry in entries.items()}
        messagebox.showinfo("Form Data", f"Submitted:\n{data}")
    
    def clear_form():
        for entry in entries.values():
            entry.delete(0, tk.END)
    
    tk.Button(button_frame, text="Submit", command=submit_form, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", command=clear_form, bg="lightcoral").pack(side="left", padx=5)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Entry Examples")
    print("=" * 30)
    print("1. Basic Entries")
    print("2. Validation Entries")
    print("3. Styled Entries")
    print("4. Special Entries")
    print("5. Event Entries")
    print("6. Formatting Entries")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_entries()
    elif choice == "2":
        validation_entries()
    elif choice == "3":
        styled_entries()
    elif choice == "4":
        special_entries()
    elif choice == "5":
        event_entries()
    elif choice == "6":
        formatting_entries()
    elif choice == "7":
        complete_entry_example()
    else:
        print("Invalid choice. Running basic entries...")
        basic_entries()

"""
Entry Properties:
------------------
- textvariable: Variable to store entry value
- width: Entry width in characters
- font: Font family, size, and style
- fg: Foreground (text) color
- bg: Background color
- show: Character to display (for passwords)
- state: "normal", "disabled", or "readonly"
- relief: Border style
- bd: Border width

Entry States:
-------------
- "normal": Default state, editable
- "disabled": Grayed out, not editable
- "readonly": Text visible but not editable

Common Show Characters:
-----------------------
- "*": Asterisks for passwords
- "•": Bullets for hidden text
- "": Empty string for normal text

Best Practices:
--------------
- Use placeholder text to guide users
- Validate input before processing
- Provide clear error messages
- Use appropriate input types
- Consider input length limits
- Test validation thoroughly

Extra Tips:
-----------
- Use textvariable for two-way data binding
- Use validate and validatecommand for real-time validation
- Use insert and delete methods to manipulate text
- Use get() to retrieve current value
- Use bind() for custom event handling
- Consider using ttk.Entry for modern styling
"""