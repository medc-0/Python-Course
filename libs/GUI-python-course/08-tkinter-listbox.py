"""
08-tkinter-listbox.py

Tkinter Listbox - Multi-item selection and display widgets

Overview:
---------
Listboxes display a list of items that users can select from. They're 
perfect for displaying data, making selections, and providing navigation 
options in your application.

Key Features:
- Single or multiple item selection
- Scrollable content for long lists
- Custom styling and colors
- Event handling for selection changes
- Dynamic content updates

Common Use Cases:
- File browsers and directory listings
- Data display and selection
- Navigation menus
- Multi-select forms
- Search result displays

Tips:
- Use scrollbars for long lists
- Provide clear selection feedback
- Use appropriate list item formatting
- Consider keyboard navigation
- Handle empty list states gracefully
"""

import tkinter as tk
from tkinter import messagebox

# Basic listbox examples
def basic_listbox():
    """Create basic listbox"""
    root = tk.Tk()
    root.title("Basic Listbox")
    root.geometry("400x300")
    
    # Create listbox
    listbox = tk.Listbox(root, height=10, width=40)
    listbox.pack(pady=10, padx=10)
    
    # Add items to listbox
    items = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6", "Item 7", "Item 8"]
    for item in items:
        listbox.insert(tk.END, item)
    
    # Submit button
    def submit_selection():
        selection = listbox.curselection()
        if selection:
            selected_items = [listbox.get(i) for i in selection]
            messagebox.showinfo("Selected Items", f"You selected: {', '.join(selected_items)}")
        else:
            messagebox.showwarning("No Selection", "Please select an item")
    
    tk.Button(root, text="Submit", command=submit_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Listbox with scrollbar
def scrolled_listbox():
    """Create listbox with scrollbar"""
    root = tk.Tk()
    root.title("Scrolled Listbox")
    root.geometry("400x300")
    
    # Create frame for listbox and scrollbar
    frame = tk.Frame(root)
    frame.pack(pady=10, padx=10)
    
    # Create listbox
    listbox = tk.Listbox(frame, height=10, width=40)
    listbox.pack(side="left", fill="both", expand=True)
    
    # Create scrollbar
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")
    
    # Configure scrollbar
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    
    # Add items to listbox
    items = [f"Item {i}" for i in range(1, 51)]  # 50 items
    for item in items:
        listbox.insert(tk.END, item)
    
    # Submit button
    def submit_selection():
        selection = listbox.curselection()
        if selection:
            selected_items = [listbox.get(i) for i in selection]
            messagebox.showinfo("Selected Items", f"You selected: {', '.join(selected_items)}")
        else:
            messagebox.showwarning("No Selection", "Please select an item")
    
    tk.Button(root, text="Submit", command=submit_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Multi-select listbox
def multi_select_listbox():
    """Create listbox with multiple selection"""
    root = tk.Tk()
    root.title("Multi-Select Listbox")
    root.geometry("400x350")
    
    # Create listbox
    listbox = tk.Listbox(root, height=10, width=40, selectmode=tk.MULTIPLE)
    listbox.pack(pady=10, padx=10)
    
    # Add items to listbox
    items = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5", "Option 6", "Option 7", "Option 8"]
    for item in items:
        listbox.insert(tk.END, item)
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def submit_selection():
        selection = listbox.curselection()
        if selection:
            selected_items = [listbox.get(i) for i in selection]
            messagebox.showinfo("Selected Items", f"You selected: {', '.join(selected_items)}")
        else:
            messagebox.showwarning("No Selection", "Please select at least one item")
    
    def select_all():
        listbox.selection_set(0, tk.END)
    
    def clear_selection():
        listbox.selection_clear(0, tk.END)
    
    tk.Button(button_frame, text="Submit", command=submit_selection, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Select All", command=select_all, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear Selection", command=clear_selection, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Styled listbox
def styled_listbox():
    """Create listbox with custom styling"""
    root = tk.Tk()
    root.title("Styled Listbox")
    root.geometry("400x350")
    
    # Create styled listbox
    listbox = tk.Listbox(root, height=10, width=40, 
                         font=("Arial", 12), fg="blue", bg="lightyellow",
                         selectbackground="lightblue", selectforeground="white")
    listbox.pack(pady=10, padx=10)
    
    # Add items to listbox
    items = ["Styled Item 1", "Styled Item 2", "Styled Item 3", "Styled Item 4", "Styled Item 5"]
    for item in items:
        listbox.insert(tk.END, item)
    
    # Submit button
    def submit_selection():
        selection = listbox.curselection()
        if selection:
            selected_items = [listbox.get(i) for i in selection]
            messagebox.showinfo("Selected Items", f"You selected: {', '.join(selected_items)}")
        else:
            messagebox.showwarning("No Selection", "Please select an item")
    
    tk.Button(root, text="Submit", command=submit_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Listbox with search functionality
def searchable_listbox():
    """Create listbox with search functionality"""
    root = tk.Tk()
    root.title("Searchable Listbox")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Search frame
    search_frame = tk.Frame(main_frame)
    search_frame.pack(fill="x", pady=(0, 10))
    
    tk.Label(search_frame, text="Search:").pack(side="left")
    search_entry = tk.Entry(search_frame, width=30)
    search_entry.pack(side="left", padx=5)
    
    # Listbox frame
    listbox_frame = tk.Frame(main_frame)
    listbox_frame.pack(fill="both", expand=True)
    
    # Create listbox
    listbox = tk.Listbox(listbox_frame, height=15, width=50)
    listbox.pack(side="left", fill="both", expand=True)
    
    # Create scrollbar
    scrollbar = tk.Scrollbar(listbox_frame)
    scrollbar.pack(side="right", fill="y")
    
    # Configure scrollbar
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    
    # Add items to listbox
    items = [f"Item {i}" for i in range(1, 101)]  # 100 items
    for item in items:
        listbox.insert(tk.END, item)
    
    # Search function
    def search_items():
        search_term = search_entry.get().lower()
        if search_term:
            # Clear previous selection
            listbox.selection_clear(0, tk.END)
            
            # Search for items
            for i in range(listbox.size()):
                item = listbox.get(i).lower()
                if search_term in item:
                    listbox.selection_set(i)
                    listbox.see(i)
                    break
    
    # Bind search to entry
    search_entry.bind("<KeyRelease>", lambda e: search_items())
    
    # Submit button
    def submit_selection():
        selection = listbox.curselection()
        if selection:
            selected_items = [listbox.get(i) for i in selection]
            messagebox.showinfo("Selected Items", f"You selected: {', '.join(selected_items)}")
        else:
            messagebox.showwarning("No Selection", "Please select an item")
    
    tk.Button(main_frame, text="Submit", command=submit_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Listbox with dynamic content
def dynamic_listbox():
    """Create listbox with dynamic content"""
    root = tk.Tk()
    root.title("Dynamic Listbox")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Listbox frame
    listbox_frame = tk.Frame(main_frame)
    listbox_frame.pack(fill="both", expand=True)
    
    # Create listbox
    listbox = tk.Listbox(listbox_frame, height=15, width=50)
    listbox.pack(side="left", fill="both", expand=True)
    
    # Create scrollbar
    scrollbar = tk.Scrollbar(listbox_frame)
    scrollbar.pack(side="right", fill="y")
    
    # Configure scrollbar
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    
    # Add initial items
    items = ["Item 1", "Item 2", "Item 3"]
    for item in items:
        listbox.insert(tk.END, item)
    
    # Add new item button
    def add_item():
        new_item = f"Item {listbox.size() + 1}"
        listbox.insert(tk.END, new_item)
    
    # Remove selected item button
    def remove_item():
        selection = listbox.curselection()
        if selection:
            listbox.delete(selection[0])
        else:
            messagebox.showwarning("No Selection", "Please select an item to remove")
    
    # Clear all items button
    def clear_all():
        listbox.delete(0, tk.END)
    
    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Add Item", command=add_item, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Remove Item", command=remove_item, bg="lightcoral").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear All", command=clear_all, bg="lightyellow").pack(side="left", padx=5)
    
    # Submit button
    def submit_selection():
        selection = listbox.curselection()
        if selection:
            selected_items = [listbox.get(i) for i in selection]
            messagebox.showinfo("Selected Items", f"You selected: {', '.join(selected_items)}")
        else:
            messagebox.showwarning("No Selection", "Please select an item")
    
    tk.Button(main_frame, text="Submit", command=submit_selection, bg="lightblue").pack(pady=10)
    
    root.mainloop()

# Complete listbox example
def complete_listbox_example():
    """Complete example showing all listbox features"""
    root = tk.Tk()
    root.title("Complete Listbox Example")
    root.geometry("600x500")
    
    # Header
    header = tk.Label(root, text="Complete Listbox Example", 
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
    
    # Left column listbox
    tk.Label(left_frame, text="Basic Listbox:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    listbox1 = tk.Listbox(left_frame, height=10, width=25)
    listbox1.pack(fill="both", expand=True, pady=5)
    
    # Add items to left listbox
    items1 = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]
    for item in items1:
        listbox1.insert(tk.END, item)
    
    # Right column listbox
    tk.Label(right_frame, text="Styled Listbox:", font=("Arial", 12, "bold")).pack(anchor="w")
    
    listbox2 = tk.Listbox(right_frame, height=10, width=25, 
                         font=("Arial", 10), fg="blue", bg="lightyellow",
                         selectbackground="lightblue", selectforeground="white")
    listbox2.pack(fill="both", expand=True, pady=5)
    
    # Add items to right listbox
    items2 = ["Styled 1", "Styled 2", "Styled 3", "Styled 4", "Styled 5"]
    for item in items2:
        listbox2.insert(tk.END, item)
    
    # Submit button
    def submit_selection():
        selection1 = listbox1.curselection()
        selection2 = listbox2.curselection()
        
        result = "Selected Items:\n"
        if selection1:
            selected_items1 = [listbox1.get(i) for i in selection1]
            result += f"Basic: {', '.join(selected_items1)}\n"
        if selection2:
            selected_items2 = [listbox2.get(i) for i in selection2]
            result += f"Styled: {', '.join(selected_items2)}"
        
        if not selection1 and not selection2:
            result = "No items selected"
        
        messagebox.showinfo("Selected Items", result)
    
    tk.Button(content_frame, text="Submit", command=submit_selection, bg="lightblue").pack(pady=20)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Listbox Examples")
    print("=" * 30)
    print("1. Basic Listbox")
    print("2. Scrolled Listbox")
    print("3. Multi-Select Listbox")
    print("4. Styled Listbox")
    print("5. Searchable Listbox")
    print("6. Dynamic Listbox")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_listbox()
    elif choice == "2":
        scrolled_listbox()
    elif choice == "3":
        multi_select_listbox()
    elif choice == "4":
        styled_listbox()
    elif choice == "5":
        searchable_listbox()
    elif choice == "6":
        dynamic_listbox()
    elif choice == "7":
        complete_listbox_example()
    else:
        print("Invalid choice. Running basic listbox...")
        basic_listbox()

"""
Listbox Properties:
-------------------
- height: Number of lines to display
- width: Number of characters per line
- font: Font family, size, and style
- fg: Foreground (text) color
- bg: Background color
- selectbackground: Background color of selected items
- selectforeground: Text color of selected items
- selectmode: Selection mode (SINGLE, MULTIPLE, EXTENDED, BROWSE)
- relief: Border style
- bd: Border width

Selection Modes:
----------------
- SINGLE: Select one item at a time
- MULTIPLE: Select multiple items with clicks
- EXTENDED: Select multiple items with Shift/Ctrl
- BROWSE: Select one item with arrow keys

Listbox Operations:
-------------------
- insert(index, item): Insert item at index
- delete(start, end): Delete items from start to end
- get(start, end): Get items from start to end
- curselection(): Get indices of selected items
- selection_set(start, end): Select items
- selection_clear(start, end): Clear selection
- see(index): Scroll to make index visible

Best Practices:
--------------
- Use scrollbars for long lists
- Provide clear selection feedback
- Use appropriate list item formatting
- Consider keyboard navigation
- Handle empty list states gracefully
- Test listbox functionality thoroughly

Extra Tips:
-----------
- Use yscrollcommand for vertical scrolling
- Use xscrollcommand for horizontal scrolling
- Use bind() for custom event handling
- Consider using ttk.Treeview for hierarchical data
- Use tags for custom item styling
- Implement search functionality for large lists
"""