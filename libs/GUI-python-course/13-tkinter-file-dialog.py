"""
13-tkinter-file-dialog.py

Tkinter File Dialog - File and directory selection dialogs

Overview:
---------
File dialogs provide a standard way for users to select files and 
directories. They're essential for applications that need to work 
with files, such as text editors, image viewers, and data processors.

Key Features:
- File selection with filters
- Directory selection
- Save and open dialogs
- Custom file types and filters
- Multiple file selection

Common Use Cases:
- Opening files for editing
- Saving files with custom names
- Selecting directories for operations
- File import/export functionality
- Batch file processing

Tips:
- Use appropriate file filters
- Provide clear dialog titles
- Handle file not found errors
- Consider file size limits
- Test with different file types
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Basic file dialogs
def basic_file_dialogs():
    """Create basic file dialogs"""
    root = tk.Tk()
    root.title("Basic File Dialogs")
    root.geometry("400x300")
    
    # Open file dialog
    def open_file():
        file_path = filedialog.askopenfilename(
            title="Open File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            messagebox.showinfo("File Selected", f"Selected file: {file_path}")
    
    # Save file dialog
    def save_file():
        file_path = filedialog.asksaveasfilename(
            title="Save File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            messagebox.showinfo("File Saved", f"File saved as: {file_path}")
    
    # Select directory dialog
    def select_directory():
        dir_path = filedialog.askdirectory(title="Select Directory")
        if dir_path:
            messagebox.showinfo("Directory Selected", f"Selected directory: {dir_path}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Open File", command=open_file, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Save File", command=save_file, bg="lightgreen").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Select Directory", command=select_directory, bg="lightyellow").pack(pady=5, fill="x")
    
    root.mainloop()

# File dialogs with filters
def filtered_file_dialogs():
    """Create file dialogs with custom filters"""
    root = tk.Tk()
    root.title("Filtered File Dialogs")
    root.geometry("400x350")
    
    # Image file dialog
    def open_image():
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            messagebox.showinfo("Image Selected", f"Selected image: {file_path}")
    
    # Document file dialog
    def open_document():
        file_path = filedialog.askopenfilename(
            title="Open Document",
            filetypes=[
                ("Document files", "*.pdf *.doc *.docx *.txt"),
                ("PDF files", "*.pdf"),
                ("Word files", "*.doc *.docx"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            messagebox.showinfo("Document Selected", f"Selected document: {file_path}")
    
    # Code file dialog
    def open_code():
        file_path = filedialog.askopenfilename(
            title="Open Code File",
            filetypes=[
                ("Python files", "*.py"),
                ("JavaScript files", "*.js"),
                ("HTML files", "*.html"),
                ("CSS files", "*.css"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            messagebox.showinfo("Code File Selected", f"Selected code file: {file_path}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Open Image", command=open_image, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Open Document", command=open_document, bg="lightgreen").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Open Code File", command=open_code, bg="lightyellow").pack(pady=5, fill="x")
    
    root.mainloop()

# Multiple file selection
def multiple_file_dialogs():
    """Create file dialogs with multiple file selection"""
    root = tk.Tk()
    root.title("Multiple File Selection")
    root.geometry("400x350")
    
    # Multiple file selection
    def select_multiple_files():
        file_paths = filedialog.askopenfilenames(
            title="Select Multiple Files",
            filetypes=[("All files", "*.*")]
        )
        if file_paths:
            file_list = "\n".join(file_paths)
            messagebox.showinfo("Files Selected", f"Selected files:\n{file_list}")
    
    # Multiple image selection
    def select_multiple_images():
        file_paths = filedialog.askopenfilenames(
            title="Select Multiple Images",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        if file_paths:
            file_list = "\n".join(file_paths)
            messagebox.showinfo("Images Selected", f"Selected images:\n{file_list}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Select Multiple Files", command=select_multiple_files, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Select Multiple Images", command=select_multiple_images, bg="lightgreen").pack(pady=5, fill="x")
    
    root.mainloop()

# File dialogs with error handling
def error_handling_file_dialogs():
    """Create file dialogs with error handling"""
    root = tk.Tk()
    root.title("File Dialogs with Error Handling")
    root.geometry("400x350")
    
    # Open file with error handling
    def open_file_safely():
        try:
            file_path = filedialog.askopenfilename(
                title="Open File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                # Check if file exists
                if os.path.exists(file_path):
                    # Check file size
                    file_size = os.path.getsize(file_path)
                    if file_size > 10 * 1024 * 1024:  # 10MB limit
                        messagebox.showwarning("File Too Large", "File is too large to open.")
                        return
                    
                    messagebox.showinfo("File Opened", f"File opened successfully: {file_path}")
                else:
                    messagebox.showerror("File Not Found", "Selected file does not exist.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    # Save file with error handling
    def save_file_safely():
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save File",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                # Check if directory exists
                dir_path = os.path.dirname(file_path)
                if not os.path.exists(dir_path):
                    messagebox.showerror("Directory Error", "Selected directory does not exist.")
                    return
                
                # Check if file already exists
                if os.path.exists(file_path):
                    result = messagebox.askyesno("File Exists", "File already exists. Do you want to overwrite it?")
                    if not result:
                        return
                
                messagebox.showinfo("File Saved", f"File saved successfully: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Open File Safely", command=open_file_safely, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Save File Safely", command=save_file_safely, bg="lightgreen").pack(pady=5, fill="x")
    
    root.mainloop()

# File dialogs with custom options
def custom_file_dialogs():
    """Create file dialogs with custom options"""
    root = tk.Tk()
    root.title("Custom File Dialogs")
    root.geometry("400x350")
    
    # Custom open dialog
    def custom_open():
        file_path = filedialog.askopenfilename(
            title="Custom Open Dialog",
            initialdir=os.path.expanduser("~"),  # Start in home directory
            filetypes=[
                ("Python files", "*.py"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            messagebox.showinfo("File Selected", f"Selected file: {file_path}")
    
    # Custom save dialog
    def custom_save():
        file_path = filedialog.asksaveasfilename(
            title="Custom Save Dialog",
            initialdir=os.path.expanduser("~/Documents"),  # Start in Documents
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("Python files", "*.py"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            messagebox.showinfo("File Saved", f"File saved as: {file_path}")
    
    # Custom directory dialog
    def custom_directory():
        dir_path = filedialog.askdirectory(
            title="Custom Directory Dialog",
            initialdir=os.path.expanduser("~")
        )
        if dir_path:
            messagebox.showinfo("Directory Selected", f"Selected directory: {dir_path}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Custom Open", command=custom_open, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Custom Save", command=custom_save, bg="lightgreen").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Custom Directory", command=custom_directory, bg="lightyellow").pack(pady=5, fill="x")
    
    root.mainloop()

# File dialogs with file operations
def file_operation_dialogs():
    """Create file dialogs with file operations"""
    root = tk.Tk()
    root.title("File Operation Dialogs")
    root.geometry("400x350")
    
    # Open and read file
    def open_and_read():
        file_path = filedialog.askopenfilename(
            title="Open File to Read",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    messagebox.showinfo("File Content", f"File content (first 100 chars):\n{content[:100]}...")
            except Exception as e:
                messagebox.showerror("Error", f"Could not read file: {e}")
    
    # Save content to file
    def save_content():
        file_path = filedialog.asksaveasfilename(
            title="Save Content to File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                content = "This is sample content to save to file."
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                messagebox.showinfo("File Saved", f"Content saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    tk.Button(button_frame, text="Open and Read", command=open_and_read, bg="lightblue").pack(pady=5, fill="x")
    tk.Button(button_frame, text="Save Content", command=save_content, bg="lightgreen").pack(pady=5, fill="x")
    
    root.mainloop()

# Complete file dialog example
def complete_file_dialog_example():
    """Complete example showing all file dialog features"""
    root = tk.Tk()
    root.title("Complete File Dialog Example")
    root.geometry("600x500")
    
    # Header
    header = tk.Label(root, text="Complete File Dialog Example", 
                     font=("Arial", 16, "bold"), bg="darkblue", fg="white")
    header.pack(fill="x", padx=10, pady=10)
    
    # Main content frame
    content_frame = tk.Frame(root, bg="lightgray")
    content_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Left column
    left_frame = tk.Frame(content_frame, bg="lightblue", relief="raised", bd=2)
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_frame, text="File Operations", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
    
    tk.Button(left_frame, text="Open File", command=lambda: filedialog.askopenfilename(), 
              bg="white").pack(pady=5, fill="x", padx=10)
    tk.Button(left_frame, text="Save File", command=lambda: filedialog.asksaveasfilename(), 
              bg="white").pack(pady=5, fill="x", padx=10)
    tk.Button(left_frame, text="Select Directory", command=lambda: filedialog.askdirectory(), 
              bg="white").pack(pady=5, fill="x", padx=10)
    
    # Right column
    right_frame = tk.Frame(content_frame, bg="lightgreen", relief="raised", bd=2)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_frame, text="File Types", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=10)
    
    tk.Button(right_frame, text="Open Image", command=lambda: filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]), 
        bg="white").pack(pady=5, fill="x", padx=10)
    tk.Button(right_frame, text="Open Document", command=lambda: filedialog.askopenfilename(
        filetypes=[("Document files", "*.pdf *.doc *.docx *.txt")]), 
        bg="white").pack(pady=5, fill="x", padx=10)
    tk.Button(right_frame, text="Open Code", command=lambda: filedialog.askopenfilename(
        filetypes=[("Code files", "*.py *.js *.html *.css")]), 
        bg="white").pack(pady=5, fill="x", padx=10)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter File Dialog Examples")
    print("=" * 35)
    print("1. Basic File Dialogs")
    print("2. Filtered File Dialogs")
    print("3. Multiple File Selection")
    print("4. Error Handling File Dialogs")
    print("5. Custom File Dialogs")
    print("6. File Operation Dialogs")
    print("7. Complete Example")
    print("=" * 35)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_file_dialogs()
    elif choice == "2":
        filtered_file_dialogs()
    elif choice == "3":
        multiple_file_dialogs()
    elif choice == "4":
        error_handling_file_dialogs()
    elif choice == "5":
        custom_file_dialogs()
    elif choice == "6":
        file_operation_dialogs()
    elif choice == "7":
        complete_file_dialog_example()
    else:
        print("Invalid choice. Running basic file dialogs...")
        basic_file_dialogs()

"""
File Dialog Types:
------------------
- askopenfilename(): Open single file
- askopenfilenames(): Open multiple files
- asksaveasfilename(): Save file
- askdirectory(): Select directory

File Dialog Options:
--------------------
- title: Dialog title
- initialdir: Initial directory
- filetypes: File type filters
- defaultextension: Default file extension
- multiple: Allow multiple selection

Common File Types:
------------------
- Text files: "*.txt"
- Image files: "*.png *.jpg *.jpeg *.gif *.bmp"
- Document files: "*.pdf *.doc *.docx"
- Code files: "*.py *.js *.html *.css"
- All files: "*.*"

Best Practices:
--------------
- Use appropriate file filters
- Provide clear dialog titles
- Handle file not found errors
- Consider file size limits
- Test with different file types
- Use error handling for file operations

Extra Tips:
-----------
- Use os.path.exists() to check file existence
- Use os.path.getsize() to check file size
- Use os.path.dirname() to get directory path
- Use os.path.expanduser() for home directory
- Use encoding parameter for text files
- Consider using pathlib for modern path handling
"""