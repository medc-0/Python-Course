"""
20-tkinter-mini-app.py

Tkinter Mini App - Complete application example

Overview:
---------
This mini application demonstrates how to combine various Tkinter 
widgets and concepts to create a functional application. It includes 
file operations, text editing, and user interface components.

Key Features:
- File operations (open, save, new)
- Text editing with formatting
- Menu bar and toolbar
- Status bar and progress indication
- Settings and preferences
- Error handling and validation

Common Use Cases:
- Text editors and notepads
- Simple document editors
- Configuration tools
- Data entry applications
- Educational and demonstration apps

Tips:
- Organize code into logical sections
- Use classes for complex applications
- Handle errors gracefully
- Provide user feedback
- Test all functionality thoroughly
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import os
import json

class MiniApp:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.create_widgets()
        self.setup_menu()
        self.setup_bindings()
        self.current_file = None
        self.settings = self.load_settings()
    
    def setup_window(self):
        """Setup the main window"""
        self.root.title("Mini App - Tkinter Example")
        self.root.geometry("800x600")
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
        # Main content frame
        self.content_frame = tk.Frame(self.root, bg="white")
        self.content_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Toolbar
        self.create_toolbar()
        
        # Text editor
        self.create_text_editor()
        
        # Status bar
        self.create_status_bar()
    
    def create_toolbar(self):
        """Create toolbar with buttons"""
        self.toolbar = tk.Frame(self.content_frame, bg="lightgray", relief="raised", bd=1)
        self.toolbar.pack(fill="x", pady=(0, 5))
        
        # File buttons
        tk.Button(self.toolbar, text="New", command=self.new_file, bg="lightblue", width=8).pack(side="left", padx=2, pady=2)
        tk.Button(self.toolbar, text="Open", command=self.open_file, bg="lightgreen", width=8).pack(side="left", padx=2, pady=2)
        tk.Button(self.toolbar, text="Save", command=self.save_file, bg="lightyellow", width=8).pack(side="left", padx=2, pady=2)
        tk.Button(self.toolbar, text="Save As", command=self.save_as_file, bg="lightcoral", width=8).pack(side="left", padx=2, pady=2)
        
        # Separator
        tk.Frame(self.toolbar, width=2, bg="gray").pack(side="left", padx=5, pady=2)
        
        # Edit buttons
        tk.Button(self.toolbar, text="Cut", command=self.cut_text, bg="lightblue", width=8).pack(side="left", padx=2, pady=2)
        tk.Button(self.toolbar, text="Copy", command=self.copy_text, bg="lightgreen", width=8).pack(side="left", padx=2, pady=2)
        tk.Button(self.toolbar, text="Paste", command=self.paste_text, bg="lightyellow", width=8).pack(side="left", padx=2, pady=2)
        tk.Button(self.toolbar, text="Clear", command=self.clear_text, bg="lightcoral", width=8).pack(side="left", padx=2, pady=2)
        
        # Separator
        tk.Frame(self.toolbar, width=2, bg="gray").pack(side="left", padx=5, pady=2)
        
        # Format buttons
        tk.Button(self.toolbar, text="Bold", command=self.toggle_bold, bg="lightblue", width=8).pack(side="left", padx=2, pady=2)
        tk.Button(self.toolbar, text="Italic", command=self.toggle_italic, bg="lightgreen", width=8).pack(side="left", padx=2, pady=2)
        tk.Button(self.toolbar, text="Underline", command=self.toggle_underline, bg="lightyellow", width=8).pack(side="left", padx=2, pady=2)
        
        # Separator
        tk.Frame(self.toolbar, width=2, bg="gray").pack(side="left", padx=5, pady=2)
        
        # Settings button
        tk.Button(self.toolbar, text="Settings", command=self.show_settings, bg="lightpink", width=8).pack(side="left", padx=2, pady=2)
    
    def create_text_editor(self):
        """Create text editor"""
        # Text editor frame
        self.text_frame = tk.Frame(self.content_frame, bg="white")
        self.text_frame.pack(expand=True, fill="both")
        
        # Text widget
        self.text_widget = scrolledtext.ScrolledText(
            self.text_frame, 
            wrap="word", 
            font=("Arial", 12),
            bg="white",
            fg="black"
        )
        self.text_widget.pack(expand=True, fill="both")
        
        # Bind text change event
        self.text_widget.bind("<KeyRelease>", self.on_text_change)
        self.text_widget.bind("<Button-1>", self.on_text_change)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Frame(self.root, bg="lightgray", relief="sunken", bd=1)
        self.status_bar.pack(side="bottom", fill="x")
        
        # Status label
        self.status_label = tk.Label(self.status_bar, text="Ready", anchor="w", bg="lightgray")
        self.status_label.pack(side="left", padx=5)
        
        # Line/column info
        self.line_col_label = tk.Label(self.status_bar, text="Line: 1, Col: 1", anchor="e", bg="lightgray")
        self.line_col_label.pack(side="right", padx=5)
    
    def setup_menu(self):
        """Setup menu bar"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        # File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        self.file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        self.file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        self.file_menu.add_command(label="Save As", command=self.save_as_file, accelerator="Ctrl+Shift+S")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Edit menu
        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=self.edit_menu)
        self.edit_menu.add_command(label="Cut", command=self.cut_text, accelerator="Ctrl+X")
        self.edit_menu.add_command(label="Copy", command=self.copy_text, accelerator="Ctrl+C")
        self.edit_menu.add_command(label="Paste", command=self.paste_text, accelerator="Ctrl+V")
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Select All", command=self.select_all, accelerator="Ctrl+A")
        self.edit_menu.add_command(label="Clear", command=self.clear_text)
        
        # Format menu
        self.format_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Format", menu=self.format_menu)
        self.format_menu.add_command(label="Bold", command=self.toggle_bold, accelerator="Ctrl+B")
        self.format_menu.add_command(label="Italic", command=self.toggle_italic, accelerator="Ctrl+I")
        self.format_menu.add_command(label="Underline", command=self.toggle_underline, accelerator="Ctrl+U")
        
        # View menu
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Settings", command=self.show_settings)
        self.view_menu.add_command(label="About", command=self.show_about)
    
    def setup_bindings(self):
        """Setup keyboard bindings"""
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-Shift-S>", lambda e: self.save_as_file())
        self.root.bind("<Control-q>", lambda e: self.root.quit())
        self.root.bind("<Control-x>", lambda e: self.cut_text())
        self.root.bind("<Control-c>", lambda e: self.copy_text())
        self.root.bind("<Control-v>", lambda e: self.paste_text())
        self.root.bind("<Control-a>", lambda e: self.select_all())
        self.root.bind("<Control-b>", lambda e: self.toggle_bold())
        self.root.bind("<Control-i>", lambda e: self.toggle_italic())
        self.root.bind("<Control-u>", lambda e: self.toggle_underline())
    
    def new_file(self):
        """Create new file"""
        if self.check_unsaved_changes():
            self.text_widget.delete("1.0", tk.END)
            self.current_file = None
            self.update_status("New file created")
    
    def open_file(self):
        """Open file"""
        if self.check_unsaved_changes():
            file_path = filedialog.askopenfilename(
                title="Open File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        self.text_widget.delete("1.0", tk.END)
                        self.text_widget.insert("1.0", content)
                        self.current_file = file_path
                        self.update_status(f"Opened: {os.path.basename(file_path)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open file: {e}")
    
    def save_file(self):
        """Save file"""
        if self.current_file:
            try:
                with open(self.current_file, 'w', encoding='utf-8') as file:
                    content = self.text_widget.get("1.0", tk.END)
                    file.write(content)
                    self.update_status(f"Saved: {os.path.basename(self.current_file)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
        else:
            self.save_as_file()
    
    def save_as_file(self):
        """Save file as"""
        file_path = filedialog.asksaveasfilename(
            title="Save File As",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    content = self.text_widget.get("1.0", tk.END)
                    file.write(content)
                    self.current_file = file_path
                    self.update_status(f"Saved as: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
    
    def cut_text(self):
        """Cut selected text"""
        try:
            self.text_widget.event_generate("<<Cut>>")
            self.update_status("Text cut")
        except:
            pass
    
    def copy_text(self):
        """Copy selected text"""
        try:
            self.text_widget.event_generate("<<Copy>>")
            self.update_status("Text copied")
        except:
            pass
    
    def paste_text(self):
        """Paste text"""
        try:
            self.text_widget.event_generate("<<Paste>>")
            self.update_status("Text pasted")
        except:
            pass
    
    def select_all(self):
        """Select all text"""
        self.text_widget.tag_add("sel", "1.0", tk.END)
        self.update_status("All text selected")
    
    def clear_text(self):
        """Clear all text"""
        if messagebox.askyesno("Clear Text", "Are you sure you want to clear all text?"):
            self.text_widget.delete("1.0", tk.END)
            self.update_status("Text cleared")
    
    def toggle_bold(self):
        """Toggle bold formatting"""
        try:
            self.text_widget.tag_add("bold", "sel.first", "sel.last")
            self.text_widget.tag_configure("bold", font=("Arial", 12, "bold"))
            self.update_status("Bold formatting applied")
        except:
            pass
    
    def toggle_italic(self):
        """Toggle italic formatting"""
        try:
            self.text_widget.tag_add("italic", "sel.first", "sel.last")
            self.text_widget.tag_configure("italic", font=("Arial", 12, "italic"))
            self.update_status("Italic formatting applied")
        except:
            pass
    
    def toggle_underline(self):
        """Toggle underline formatting"""
        try:
            self.text_widget.tag_add("underline", "sel.first", "sel.last")
            self.text_widget.tag_configure("underline", underline=True)
            self.update_status("Underline formatting applied")
        except:
            pass
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        
        # Center the settings window
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Settings content
        main_frame = tk.Frame(settings_window, padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")
        
        # Font settings
        font_frame = tk.LabelFrame(main_frame, text="Font Settings", padx=10, pady=10)
        font_frame.pack(fill="x", pady=10)
        
        tk.Label(font_frame, text="Font Family:").pack(anchor="w")
        font_family = tk.StringVar(value=self.settings.get("font_family", "Arial"))
        font_combo = ttk.Combobox(font_frame, textvariable=font_family, values=["Arial", "Times", "Courier", "Helvetica"])
        font_combo.pack(fill="x", pady=5)
        
        tk.Label(font_frame, text="Font Size:").pack(anchor="w")
        font_size = tk.IntVar(value=self.settings.get("font_size", 12))
        font_scale = tk.Scale(font_frame, from_=8, to=24, orient="horizontal", variable=font_size)
        font_scale.pack(fill="x", pady=5)
        
        # Color settings
        color_frame = tk.LabelFrame(main_frame, text="Color Settings", padx=10, pady=10)
        color_frame.pack(fill="x", pady=10)
        
        tk.Label(color_frame, text="Background Color:").pack(anchor="w")
        bg_color = tk.StringVar(value=self.settings.get("bg_color", "white"))
        bg_combo = ttk.Combobox(color_frame, textvariable=bg_color, values=["white", "lightgray", "lightblue", "lightgreen"])
        bg_combo.pack(fill="x", pady=5)
        
        tk.Label(color_frame, text="Text Color:").pack(anchor="w")
        fg_color = tk.StringVar(value=self.settings.get("fg_color", "black"))
        fg_combo = ttk.Combobox(color_frame, textvariable=fg_color, values=["black", "blue", "red", "green", "purple"])
        fg_combo.pack(fill="x", pady=5)
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill="x", pady=20)
        
        def apply_settings():
            self.settings["font_family"] = font_family.get()
            self.settings["font_size"] = font_size.get()
            self.settings["bg_color"] = bg_color.get()
            self.settings["fg_color"] = fg_color.get()
            
            # Apply settings to text widget
            self.text_widget.config(
                font=(font_family.get(), font_size.get()),
                bg=bg_color.get(),
                fg=fg_color.get()
            )
            
            self.save_settings()
            settings_window.destroy()
            self.update_status("Settings applied")
        
        def cancel_settings():
            settings_window.destroy()
        
        tk.Button(button_frame, text="Apply", command=apply_settings, bg="lightgreen").pack(side="left", padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel_settings, bg="lightcoral").pack(side="left", padx=5)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", "Mini App - Tkinter Example\n\nA demonstration of Tkinter capabilities including:\n• File operations\n• Text editing\n• Menu system\n• Settings\n• Error handling")
    
    def check_unsaved_changes(self):
        """Check for unsaved changes"""
        # This is a simplified check - in a real app, you'd track changes
        return True
    
    def on_text_change(self, event=None):
        """Handle text change events"""
        # Update line/column info
        cursor_pos = self.text_widget.index(tk.INSERT)
        line, col = cursor_pos.split('.')
        self.line_col_label.config(text=f"Line: {line}, Col: {int(col)+1}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.after(3000, lambda: self.status_label.config(text="Ready"))
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", 'r') as file:
                    return json.load(file)
        except:
            pass
        return {"font_family": "Arial", "font_size": 12, "bg_color": "white", "fg_color": "black"}
    
    def save_settings(self):
        """Save settings to file"""
        try:
            with open("settings.json", 'w') as file:
                json.dump(self.settings, file)
        except:
            pass
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

# Main execution
if __name__ == "__main__":
    app = MiniApp()
    app.run()

"""
Mini App Features:
------------------
- File operations (new, open, save, save as)
- Text editing with formatting
- Menu bar with keyboard shortcuts
- Toolbar with common actions
- Status bar with information
- Settings dialog
- Error handling and validation

Application Structure:
---------------------
- Class-based design for organization
- Separation of concerns
- Event handling and bindings
- Settings persistence
- User feedback and status updates

Best Practices:
--------------
- Organize code into logical sections
- Use classes for complex applications
- Handle errors gracefully
- Provide user feedback
- Test all functionality thoroughly
- Use consistent styling
- Consider accessibility

Extra Tips:
-----------
- Use threading for long-running operations
- Implement auto-save functionality
- Add undo/redo capabilities
- Use configuration files for settings
- Implement plugin architecture
- Add logging for debugging
- Consider using MVC pattern for larger apps
"""