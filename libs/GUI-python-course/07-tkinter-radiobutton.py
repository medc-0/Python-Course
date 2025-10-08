"""
07-tkinter-radiobutton.py

Tkinter Radio Buttons - Single selection widgets for user choices

Overview:
---------
Radio buttons allow users to select exactly one option from a group. 
They're perfect for mutually exclusive choices like gender, age ranges, 
or preference settings.

Key Features:
- Single selection from a group
- Custom styling and colors
- Disabled/enabled states
- Event handling for selection changes
- Grouped radio buttons for related options

Common Use Cases:
- Gender selection (Male, Female, Other)
- Age range selection
- Preference settings
- Single-choice forms
- Option selection menus

Tips:
- Use clear, descriptive labels
- Group related radio buttons together
- Provide visual feedback for selection changes
- Use consistent styling across your app
- Consider keyboard accessibility
"""

import tkinter as tk
from tkinter import messagebox

# Basic radio button examples
def basic_radio_buttons():
    """Create basic radio buttons"""
    root = tk.Tk()
    root.title("Basic Radio Buttons")
    root.geometry("400x300")
    
    # Create radio buttons
    var = tk.StringVar()
    var.set("option1")  # Set default selection
    
    rb1 = tk.Radiobutton(root, text="Option 1", variable=var, value="option1")
    rb1.pack(pady=5, anchor="w")
    
    rb2 = tk.Radiobutton(root, text="Option 2", variable=var, value="option2")
    rb2.pack(pady=5, anchor="w")
    
    rb3 = tk.Radiobutton(root, text="Option 3", variable=var, value="option3")
    rb3.pack(pady=5, anchor="w")
    
    # Submit button
    def submit_choice():
        selected = var.get()
        messagebox.showinfo("Selected Option", f"You selected: {selected}")
    
    tk.Button(root, text="Submit", command=submit_choice, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Radio button with event handling
def event_radio_buttons():
    """Create radio buttons with event handling"""
    root = tk.Tk()
    root.title("Event Handling Radio Buttons")
    root.geometry("400x350")
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 12))
    status_label.pack(pady=10)
    
    # Radio button variable
    var = tk.StringVar()
    var.set("option1")
    
    # Event handler
    def on_radio_change():
        selected = var.get()
        status_label.config(text=f"Selected: {selected}", fg="green")
    
    # Create radio buttons with event binding
    rb1 = tk.Radiobutton(root, text="Option 1", variable=var, value="option1", command=on_radio_change)
    rb1.pack(pady=5, anchor="w")
    
    rb2 = tk.Radiobutton(root, text="Option 2", variable=var, value="option2", command=on_radio_change)
    rb2.pack(pady=5, anchor="w")
    
    rb3 = tk.Radiobutton(root, text="Option 3", variable=var, value="option3", command=on_radio_change)
    rb3.pack(pady=5, anchor="w")
    
    # Reset button
    def reset_selection():
        var.set("option1")
        on_radio_change()
    
    tk.Button(root, text="Reset", command=reset_selection, bg="lightcoral").pack(pady=10)
    
    root.mainloop()

# Styled radio buttons
def styled_radio_buttons():
    """Create radio buttons with custom styling"""
    root = tk.Tk()
    root.title("Styled Radio Buttons")
    root.geometry("400x350")
    
    # Create styled radio buttons
    var = tk.StringVar()
    var.set("option1")
    
    # Styled radio button 1
    rb1 = tk.Radiobutton(root, text="Bold Option", variable=var, value="option1",
                         font=("Arial", 12, "bold"), fg="blue", bg="lightblue")
    rb1.pack(pady=5, anchor="w", padx=10)
    
    # Styled radio button 2
    rb2 = tk.Radiobutton(root, text="Italic Option", variable=var, value="option2",
                         font=("Arial", 12, "italic"), fg="green", bg="lightgreen")
    rb2.pack(pady=5, anchor="w", padx=10)
    
    # Styled radio button 3
    rb3 = tk.Radiobutton(root, text="Underlined Option", variable=var, value="option3",
                         font=("Arial", 12, "underline"), fg="red", bg="lightcoral")
    rb3.pack(pady=5, anchor="w", padx=10)
    
    # Submit button
    def submit_choice():
        selected = var.get()
        messagebox.showinfo("Selected Option", f"You selected: {selected}")
    
    tk.Button(root, text="Submit", command=submit_choice, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Grouped radio buttons
def grouped_radio_buttons():
    """Create grouped radio buttons for related options"""
    root = tk.Tk()
    root.title("Grouped Radio Buttons")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Gender selection group
    gender_frame = tk.LabelFrame(main_frame, text="Gender", font=("Arial", 12, "bold"))
    gender_frame.pack(fill="x", pady=10)
    
    gender_var = tk.StringVar()
    gender_var.set("male")
    
    gender_options = [("Male", "male"), ("Female", "female"), ("Other", "other")]
    
    for text, value in gender_options:
        rb = tk.Radiobutton(gender_frame, text=text, variable=gender_var, value=value)
        rb.pack(anchor="w", padx=10, pady=2)
    
    # Age range group
    age_frame = tk.LabelFrame(main_frame, text="Age Range", font=("Arial", 12, "bold"))
    age_frame.pack(fill="x", pady=10)
    
    age_var = tk.StringVar()
    age_var.set("18-25")
    
    age_options = [("18-25", "18-25"), ("26-35", "26-35"), ("36-45", "36-45"), ("46+", "46+")]
    
    for text, value in age_options:
        rb = tk.Radiobutton(age_frame, text=text, variable=age_var, value=value)
        rb.pack(anchor="w", padx=10, pady=2)
    
    # Submit button
    def submit_choices():
        gender = gender_var.get()
        age = age_var.get()
        messagebox.showinfo("Selected Options", f"Gender: {gender}\nAge Range: {age}")
    
    tk.Button(main_frame, text="Submit", command=submit_choices, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Radio button with validation
def validation_radio_buttons():
    """Create radio buttons with validation"""
    root = tk.Tk()
    root.title("Validation Radio Buttons")
    root.geometry("400x350")
    
    # Create radio buttons
    var = tk.StringVar()
    var.set("option1")
    
    rb1 = tk.Radiobutton(root, text="I agree to the terms and conditions", variable=var, value="agree")
    rb1.pack(pady=5, anchor="w")
    
    rb2 = tk.Radiobutton(root, text="I disagree with the terms and conditions", variable=var, value="disagree")
    rb2.pack(pady=5, anchor="w")
    
    # Validation function
    def validate_choice():
        selected = var.get()
        
        if selected == "agree":
            messagebox.showinfo("Success", "Thank you for agreeing to the terms!")
        elif selected == "disagree":
            messagebox.showwarning("Warning", "You must agree to the terms to continue.")
        else:
            messagebox.showerror("Error", "Please make a selection.")
    
    # Submit button
    tk.Button(root, text="Submit", command=validate_choice, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Radio button with dynamic content
def dynamic_radio_buttons():
    """Create radio buttons with dynamic content"""
    root = tk.Tk()
    root.title("Dynamic Radio Buttons")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Options list
    options = ["Option 1", "Option 2", "Option 3"]
    radio_vars = []
    radio_buttons = []
    
    # Create radio buttons dynamically
    var = tk.StringVar()
    var.set("option1")
    
    for i, option in enumerate(options):
        value = f"option{i+1}"
        rb = tk.Radiobutton(main_frame, text=option, variable=var, value=value)
        rb.pack(anchor="w", padx=10, pady=2)
        radio_buttons.append(rb)
    
    # Add new option button
    def add_option():
        new_option = f"Option {len(options) + 1}"
        options.append(new_option)
        
        value = f"option{len(options)}"
        rb = tk.Radiobutton(main_frame, text=new_option, variable=var, value=value)
        rb.pack(anchor="w", padx=10, pady=2)
        radio_buttons.append(rb)
    
    # Remove last option button
    def remove_option():
        if len(options) > 1:
            options.pop()
            radio_buttons.pop().destroy()
    
    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Add Option", command=add_option, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Remove Option", command=remove_option, bg="lightcoral").pack(side="left", padx=5)
    
    # Submit button
    def submit_choice():
        selected = var.get()
        messagebox.showinfo("Selected Option", f"You selected: {selected}")
    
    tk.Button(main_frame, text="Submit", command=submit_choice, bg="lightblue").pack(pady=20)
    
    root.mainloop()

# Complete radio button example
def complete_radio_example():
    """Complete example showing all radio button features"""
    root = tk.Tk()
    root.title("Complete Radio Button Example")
    root.geometry("600x500")
    
    # Header
    header = tk.Label(root, text="Complete Radio Button Example", 
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
    
    # Left column radio buttons
    tk.Label(left_frame, text="Basic Options:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    var1 = tk.StringVar()
    var1.set("option1")
    
    tk.Radiobutton(left_frame, text="Option 1", variable=var1, value="option1").pack(anchor="w", pady=2)
    tk.Radiobutton(left_frame, text="Option 2", variable=var1, value="option2").pack(anchor="w", pady=2)
    tk.Radiobutton(left_frame, text="Option 3", variable=var1, value="option3").pack(anchor="w", pady=2)
    
    # Right column radio buttons
    tk.Label(right_frame, text="Styled Options:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    var2 = tk.StringVar()
    var2.set("styled1")
    
    tk.Radiobutton(right_frame, text="Bold Option", variable=var2, value="styled1", 
                  font=("Arial", 10, "bold"), fg="blue").pack(anchor="w", pady=2)
    tk.Radiobutton(right_frame, text="Italic Option", variable=var2, value="styled2", 
                  font=("Arial", 10, "italic"), fg="green").pack(anchor="w", pady=2)
    tk.Radiobutton(right_frame, text="Underlined Option", variable=var2, value="styled3", 
                  font=("Arial", 10, "underline"), fg="red").pack(anchor="w", pady=2)
    
    # Submit button
    def submit_choices():
        basic_selected = var1.get()
        styled_selected = var2.get()
        messagebox.showinfo("Selected Options", f"Basic: {basic_selected}\nStyled: {styled_selected}")
    
    tk.Button(content_frame, text="Submit", command=submit_choices, bg="lightblue").pack(pady=20)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Radio Button Examples")
    print("=" * 35)
    print("1. Basic Radio Buttons")
    print("2. Event Handling Radio Buttons")
    print("3. Styled Radio Buttons")
    print("4. Grouped Radio Buttons")
    print("5. Validation Radio Buttons")
    print("6. Dynamic Radio Buttons")
    print("7. Complete Example")
    print("=" * 35)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_radio_buttons()
    elif choice == "2":
        event_radio_buttons()
    elif choice == "3":
        styled_radio_buttons()
    elif choice == "4":
        grouped_radio_buttons()
    elif choice == "5":
        validation_radio_buttons()
    elif choice == "6":
        dynamic_radio_buttons()
    elif choice == "7":
        complete_radio_example()
    else:
        print("Invalid choice. Running basic radio buttons...")
        basic_radio_buttons()

"""
Radio Button Properties:
------------------------
- text: Radio button label text
- variable: StringVar or IntVar to store selection
- value: Value to store when selected
- command: Function to call when selection changes
- font: Font family, size, and style
- fg: Foreground (text) color
- bg: Background color
- state: "normal", "disabled", or "active"
- relief: Border style
- bd: Border width

Radio Button States:
--------------------
- "normal": Default state, clickable
- "disabled": Grayed out, not clickable
- "active": Currently being pressed

Common Use Cases:
-----------------
- Gender selection (Male, Female, Other)
- Age range selection
- Preference settings
- Single-choice forms
- Option selection menus

Best Practices:
--------------
- Use clear, descriptive labels
- Group related radio buttons together
- Provide visual feedback for selection changes
- Use consistent styling throughout your app
- Consider keyboard accessibility
- Test radio button functionality thoroughly

Extra Tips:
-----------
- Use StringVar or IntVar for two-way data binding
- Use command parameter for real-time updates
- Use state parameter to disable radio buttons
- Use relief and borderwidth for custom borders
- Consider using ttk.Radiobutton for modern styling
- Use compound for image and text positioning
"""