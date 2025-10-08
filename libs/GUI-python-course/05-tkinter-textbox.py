"""
05-tkinter-textbox.py

Tkinter Text Widget - Multi-line text input and display

Overview:
---------
The Text widget is perfect for multi-line text input, text editing, and 
displaying formatted text. It's more powerful than Entry widgets and 
supports rich text formatting.

Key Features:
- Multi-line text input and display
- Text formatting (bold, italic, colors)
- Text selection and editing
- Scrollbars for long content
- Text tags for styling

Common Use Cases:
- Text editors and notepads
- Chat applications and messages
- Code editors and syntax highlighting
- Rich text documents
- Log displays and output areas

Tips:
- Use scrollbars for long content
- Implement text formatting with tags
- Handle text selection and editing
- Use appropriate font families for code
- Consider text wrapping for readability
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext

# Basic text widget
def basic_text_widget():
    """Create basic text widget"""
    root = tk.Tk()
    root.title("Basic Text Widget")
    root.geometry("500x400")
    
    # Simple text widget
    text_widget = tk.Text(root, height=15, width=50)
    text_widget.pack(pady=10, padx=10)
    
    # Insert some text
    text_widget.insert("1.0", "This is a basic text widget.\n")
    text_widget.insert("2.0", "You can type multiple lines here.\n")
    text_widget.insert("3.0", "It supports text editing and formatting.\n")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def get_text():
        content = text_widget.get("1.0", tk.END)
        messagebox.showinfo("Text Content", f"Text content:\n{content}")
    
    def clear_text():
        text_widget.delete("1.0", tk.END)
    
    tk.Button(button_frame, text="Get Text", command=get_text, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", command=clear_text, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Text widget with scrollbar
def scrolled_text_widget():
    """Create text widget with scrollbar"""
    root = tk.Tk()
    root.title("Scrolled Text Widget")
    root.geometry("500x400")
    
    # Create scrolled text widget
    text_widget = scrolledtext.ScrolledText(root, height=15, width=50)
    text_widget.pack(pady=10, padx=10)
    
    # Insert long text
    long_text = """This is a long text that will demonstrate the scrollbar functionality.
    
You can scroll up and down to see all the content.
The scrollbar appears automatically when needed.

This text widget supports:
- Multi-line text input
- Text selection and editing
- Copy, cut, and paste operations
- Undo and redo functionality
- Text formatting and styling

You can type as much text as you want, and the scrollbar will handle the overflow.
This is useful for text editors, chat applications, and log displays.

The ScrolledText widget combines a Text widget with a Scrollbar widget for convenience.
It's perfect for applications that need to display or edit large amounts of text.

You can also programmatically insert text, delete text, and manipulate the content.
The widget supports various text operations and can be customized with different fonts and colors."""
    
    text_widget.insert("1.0", long_text)
    
    root.mainloop()

# Text widget with formatting
def formatted_text_widget():
    """Create text widget with text formatting"""
    root = tk.Tk()
    root.title("Formatted Text Widget")
    root.geometry("600x500")
    
    # Create text widget
    text_widget = tk.Text(root, height=20, width=60, font=("Arial", 12))
    text_widget.pack(pady=10, padx=10)
    
    # Insert formatted text
    text_widget.insert("1.0", "Formatted Text Example\n\n")
    
    # Configure tags for formatting
    text_widget.tag_configure("title", font=("Arial", 16, "bold"), foreground="blue")
    text_widget.tag_configure("subtitle", font=("Arial", 14, "bold"), foreground="green")
    text_widget.tag_configure("code", font=("Courier", 10), background="lightgray")
    text_widget.tag_configure("highlight", background="yellow")
    text_widget.tag_configure("error", foreground="red", font=("Arial", 12, "bold"))
    
    # Insert formatted content
    text_widget.insert("3.0", "This is a title\n", "title")
    text_widget.insert("4.0", "This is a subtitle\n", "subtitle")
    text_widget.insert("5.0", "This is normal text.\n")
    text_widget.insert("6.0", "This is highlighted text.\n", "highlight")
    text_widget.insert("7.0", "This is error text.\n", "error")
    text_widget.insert("8.0", "This is code: print('Hello, World!')\n", "code")
    
    # Buttons for formatting
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def make_bold():
        text_widget.tag_add("bold", "sel.first", "sel.last")
        text_widget.tag_configure("bold", font=("Arial", 12, "bold"))
    
    def make_italic():
        text_widget.tag_add("italic", "sel.first", "sel.last")
        text_widget.tag_configure("italic", font=("Arial", 12, "italic"))
    
    def make_highlight():
        text_widget.tag_add("highlight", "sel.first", "sel.last")
    
    tk.Button(button_frame, text="Bold", command=make_bold, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Italic", command=make_italic, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Highlight", command=make_highlight, bg="yellow").pack(side="left", padx=5)
    
    root.mainloop()

# Text widget with line numbers
def line_numbered_text():
    """Create text widget with line numbers"""
    root = tk.Tk()
    root.title("Text Widget with Line Numbers")
    root.geometry("600x500")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create line number widget
    line_numbers = tk.Text(main_frame, width=4, height=20, font=("Courier", 10), 
                          bg="lightgray", state="disabled")
    line_numbers.pack(side="left", fill="y")
    
    # Create main text widget
    text_widget = tk.Text(main_frame, height=20, width=60, font=("Courier", 10))
    text_widget.pack(side="left", fill="both", expand=True)
    
    # Update line numbers
    def update_line_numbers():
        line_numbers.config(state="normal")
        line_numbers.delete("1.0", tk.END)
        
        line_count = int(text_widget.index("end-1c").split(".")[0])
        for i in range(1, line_count + 1):
            line_numbers.insert(f"{i}.0", f"{i:3d}\n")
        
        line_numbers.config(state="disabled")
    
    # Bind events to update line numbers
    text_widget.bind("<KeyRelease>", lambda e: update_line_numbers())
    text_widget.bind("<Button-1>", lambda e: update_line_numbers())
    
    # Insert some sample text
    sample_text = """def hello_world():
    print("Hello, World!")
    return "Success"

class Example:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

if __name__ == "__main__":
    obj = Example()
    result = hello_world()
    print(f"Value: {obj.get_value()}")
    print(f"Result: {result}")"""
    
    text_widget.insert("1.0", sample_text)
    update_line_numbers()
    
    root.mainloop()

# Text widget with search functionality
def searchable_text_widget():
    """Create text widget with search functionality"""
    root = tk.Tk()
    root.title("Searchable Text Widget")
    root.geometry("600x500")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Search frame
    search_frame = tk.Frame(main_frame)
    search_frame.pack(fill="x", pady=(0, 10))
    
    tk.Label(search_frame, text="Search:").pack(side="left")
    search_entry = tk.Entry(search_frame, width=30)
    search_entry.pack(side="left", padx=5)
    
    def search_text():
        search_term = search_entry.get()
        if search_term:
            # Clear previous highlights
            text_widget.tag_remove("search", "1.0", tk.END)
            
            # Search for text
            start = "1.0"
            while True:
                pos = text_widget.search(search_term, start, tk.END)
                if not pos:
                    break
                end = f"{pos}+{len(search_term)}c"
                text_widget.tag_add("search", pos, end)
                start = end
            
            # Configure search highlight
            text_widget.tag_configure("search", background="yellow")
    
    tk.Button(search_frame, text="Search", command=search_text, bg="lightblue").pack(side="left", padx=5)
    
    # Create text widget
    text_widget = tk.Text(main_frame, height=20, width=60, font=("Arial", 12))
    text_widget.pack(fill="both", expand=True)
    
    # Insert sample text
    sample_text = """This is a searchable text widget.
You can search for specific words or phrases.
The search functionality will highlight matching text.
Try searching for words like 'search', 'text', or 'widget'.
The highlighting will show all occurrences of your search term.
You can search for partial words or complete phrases.
The search is case-sensitive and will find exact matches."""
    
    text_widget.insert("1.0", sample_text)
    
    root.mainloop()

# Text widget with syntax highlighting
def syntax_highlighted_text():
    """Create text widget with basic syntax highlighting"""
    root = tk.Tk()
    root.title("Syntax Highlighted Text Widget")
    root.geometry("600x500")
    
    # Create text widget
    text_widget = tk.Text(root, height=20, width=60, font=("Courier", 10))
    text_widget.pack(pady=10, padx=10)
    
    # Configure syntax highlighting tags
    text_widget.tag_configure("keyword", foreground="blue", font=("Courier", 10, "bold"))
    text_widget.tag_configure("string", foreground="green")
    text_widget.tag_configure("comment", foreground="gray", font=("Courier", 10, "italic"))
    text_widget.tag_configure("number", foreground="red")
    
    # Insert Python code
    python_code = """# This is a Python code example
def hello_world():
    \"\"\"This function prints hello world\"\"\"
    message = "Hello, World!"
    print(message)
    return message

class Example:
    def __init__(self, value=42):
        self.value = value
        self.name = "example"
    
    def get_value(self):
        return self.value
    
    def set_value(self, new_value):
        self.value = new_value

if __name__ == "__main__":
    obj = Example()
    result = hello_world()
    print(f"Value: {obj.get_value()}")
    print(f"Result: {result}")"""
    
    text_widget.insert("1.0", python_code)
    
    # Apply syntax highlighting
    def apply_syntax_highlighting():
        # Keywords
        keywords = ["def", "class", "if", "else", "elif", "for", "while", "return", "import", "from"]
        for keyword in keywords:
            start = "1.0"
            while True:
                pos = text_widget.search(keyword, start, tk.END)
                if not pos:
                    break
                end = f"{pos}+{len(keyword)}c"
                text_widget.tag_add("keyword", pos, end)
                start = end
        
        # Strings
        start = "1.0"
        while True:
            pos = text_widget.search('"', start, tk.END)
            if not pos:
                break
            end = text_widget.search('"', f"{pos}+1c", tk.END)
            if end:
                text_widget.tag_add("string", pos, f"{end}+1c")
                start = f"{end}+1c"
            else:
                break
        
        # Comments
        start = "1.0"
        while True:
            pos = text_widget.search("#", start, tk.END)
            if not pos:
                break
            line_end = text_widget.index(f"{pos} lineend")
            text_widget.tag_add("comment", pos, line_end)
            start = line_end
    
    apply_syntax_highlighting()
    
    root.mainloop()

# Complete text widget example
def complete_text_example():
    """Complete example showing all text widget features"""
    root = tk.Tk()
    root.title("Complete Text Widget Example")
    root.geometry("700x600")
    
    # Header
    header = tk.Label(root, text="Complete Text Widget Example", 
                   font=("Arial", 16, "bold"))
    header.pack(pady=10)
    
    # Main content frame
    content_frame = tk.Frame(root)
    content_frame.pack(expand=True, fill="both", padx=20, pady=10)
    
    # Create scrolled text widget
    text_widget = scrolledtext.ScrolledText(content_frame, height=20, width=70, 
                                          font=("Arial", 11))
    text_widget.pack(fill="both", expand=True)
    
    # Insert sample content
    sample_content = """Welcome to the Complete Text Widget Example!

This text widget demonstrates various features:

1. Multi-line text input and display
2. Text formatting and styling
3. Text selection and editing
4. Scrollbars for long content
5. Text tags for custom styling

You can:
- Type and edit text
- Select text with mouse or keyboard
- Copy, cut, and paste text
- Use keyboard shortcuts (Ctrl+C, Ctrl+V, etc.)
- Format text with different styles

Try selecting some text and using the formatting buttons below!

This is a long text that will demonstrate the scrollbar functionality.
You can scroll up and down to see all the content.
The scrollbar appears automatically when needed.

The text widget supports various text operations and can be customized
with different fonts, colors, and styles. It's perfect for text editors,
chat applications, and any application that needs to display or edit text."""
    
    text_widget.insert("1.0", sample_content)
    
    # Formatting buttons
    button_frame = tk.Frame(content_frame)
    button_frame.pack(pady=10)
    
    def make_bold():
        text_widget.tag_add("bold", "sel.first", "sel.last")
        text_widget.tag_configure("bold", font=("Arial", 11, "bold"))
    
    def make_italic():
        text_widget.tag_add("italic", "sel.first", "sel.last")
        text_widget.tag_configure("italic", font=("Arial", 11, "italic"))
    
    def make_highlight():
        text_widget.tag_add("highlight", "sel.first", "sel.last")
        text_widget.tag_configure("highlight", background="yellow")
    
    def clear_formatting():
        text_widget.tag_remove("bold", "sel.first", "sel.last")
        text_widget.tag_remove("italic", "sel.first", "sel.last")
        text_widget.tag_remove("highlight", "sel.first", "sel.last")
    
    tk.Button(button_frame, text="Bold", command=make_bold, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Italic", command=make_italic, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Highlight", command=make_highlight, bg="yellow").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear Format", command=clear_formatting, bg="lightcoral").pack(side="left", padx=5)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Text Widget Examples")
    print("=" * 35)
    print("1. Basic Text Widget")
    print("2. Scrolled Text Widget")
    print("3. Formatted Text Widget")
    print("4. Line Numbered Text")
    print("5. Searchable Text Widget")
    print("6. Syntax Highlighted Text")
    print("7. Complete Example")
    print("=" * 35)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_text_widget()
    elif choice == "2":
        scrolled_text_widget()
    elif choice == "3":
        formatted_text_widget()
    elif choice == "4":
        line_numbered_text()
    elif choice == "5":
        searchable_text_widget()
    elif choice == "6":
        syntax_highlighted_text()
    elif choice == "7":
        complete_text_example()
    else:
        print("Invalid choice. Running basic text widget...")
        basic_text_widget()

"""
Text Widget Properties:
-----------------------
- height: Number of lines to display
- width: Number of characters per line
- font: Font family, size, and style
- fg: Foreground (text) color
- bg: Background color
- wrap: Text wrapping mode
- state: "normal", "disabled", or "readonly"
- relief: Border style
- bd: Border width

Text Operations:
----------------
- insert(index, text): Insert text at index
- delete(start, end): Delete text from start to end
- get(start, end): Get text from start to end
- see(index): Scroll to make index visible
- mark_set(name, index): Set a mark at index
- mark_unset(name): Remove a mark

Text Tags:
----------
- tag_configure(name, **options): Configure tag appearance
- tag_add(name, start, end): Apply tag to text range
- tag_remove(name, start, end): Remove tag from text range
- tag_delete(name): Delete a tag

Best Practices:
--------------
- Use scrollbars for long content
- Implement text formatting with tags
- Handle text selection and editing
- Use appropriate font families for code
- Consider text wrapping for readability
- Test text operations thoroughly

Extra Tips:
-----------
- Use textvariable for two-way data binding
- Use marks and tags for advanced text manipulation
- Use bind() for custom event handling
- Consider using ttk.Text for modern styling
- Use ScrolledText for automatic scrollbars
- Implement undo/redo functionality for better UX
"""