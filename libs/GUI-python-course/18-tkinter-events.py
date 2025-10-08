"""
18-tkinter-events.py

Tkinter Events - Event handling and user interactions

Overview:
---------
Events are the foundation of interactive GUI applications. They handle 
user interactions like mouse clicks, keyboard input, and window events. 
Understanding events is crucial for creating responsive applications.

Key Features:
- Mouse events (clicks, movement, scrolling)
- Keyboard events (key presses, releases)
- Window events (resize, focus, close)
- Custom event handling
- Event binding and unbinding

Common Use Cases:
- Interactive drawing and painting
- Form validation and input handling
- Game controls and navigation
- Custom widget behavior
- User interface responsiveness

Tips:
- Use appropriate event types for different interactions
- Handle events efficiently to avoid performance issues
- Provide visual feedback for user actions
- Consider accessibility and keyboard navigation
- Test event handling with different input devices
"""

import tkinter as tk
from tkinter import messagebox

# Basic mouse events
def basic_mouse_events():
    """Create basic mouse event handling"""
    root = tk.Tk()
    root.title("Basic Mouse Events")
    root.geometry("500x400")
    
    # Create canvas
    canvas = tk.Canvas(root, width=400, height=300, bg="white")
    canvas.pack(pady=20)
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 12))
    status_label.pack(pady=10)
    
    # Mouse event handlers
    def on_mouse_click(event):
        status_label.config(text=f"Mouse clicked at ({event.x}, {event.y})")
        canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill="red", outline="darkred", width=2)
    
    def on_mouse_drag(event):
        status_label.config(text=f"Mouse dragged to ({event.x}, {event.y})")
        canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="blue", outline="darkblue", width=1)
    
    def on_mouse_release(event):
        status_label.config(text=f"Mouse released at ({event.x}, {event.y})")
    
    def on_mouse_motion(event):
        status_label.config(text=f"Mouse at ({event.x}, {event.y})")
    
    # Bind mouse events
    canvas.bind("<Button-1>", on_mouse_click)  # Left click
    canvas.bind("<B1-Motion>", on_mouse_drag)  # Left drag
    canvas.bind("<ButtonRelease-1>", on_mouse_release)  # Left release
    canvas.bind("<Motion>", on_mouse_motion)  # Mouse movement
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def clear_canvas():
        canvas.delete("all")
        status_label.config(text="Canvas cleared")
    
    tk.Button(button_frame, text="Clear", command=clear_canvas, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Keyboard events
def keyboard_events():
    """Create keyboard event handling"""
    root = tk.Tk()
    root.title("Keyboard Events")
    root.geometry("500x400")
    
    # Create text widget
    text_widget = tk.Text(root, height=15, width=50, wrap="word")
    text_widget.pack(pady=20)
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 12))
    status_label.pack(pady=10)
    
    # Keyboard event handlers
    def on_key_press(event):
        status_label.config(text=f"Key pressed: {event.keysym}")
    
    def on_key_release(event):
        status_label.config(text=f"Key released: {event.keysym}")
    
    def on_text_input(event):
        status_label.config(text=f"Text input: {event.char}")
    
    # Bind keyboard events
    text_widget.bind("<KeyPress>", on_key_press)
    text_widget.bind("<KeyRelease>", on_key_release)
    text_widget.bind("<Key>", on_text_input)
    
    # Focus the text widget
    text_widget.focus_set()
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def clear_text():
        text_widget.delete("1.0", tk.END)
        status_label.config(text="Text cleared")
    
    tk.Button(button_frame, text="Clear", command=clear_text, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Window events
def window_events():
    """Create window event handling"""
    root = tk.Tk()
    root.title("Window Events")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Status label
    status_label = tk.Label(main_frame, text="Status: Ready", font=("Arial", 12), bg="lightgray")
    status_label.pack(pady=10)
    
    # Window event handlers
    def on_window_resize(event):
        status_label.config(text=f"Window resized to {event.width}x{event.height}")
    
    def on_window_focus(event):
        status_label.config(text="Window gained focus")
    
    def on_window_unfocus(event):
        status_label.config(text="Window lost focus")
    
    def on_window_close(event):
        if messagebox.askyesno("Exit", "Do you want to exit?"):
            root.destroy()
        else:
            return "break"  # Prevent window from closing
    
    # Bind window events
    root.bind("<Configure>", on_window_resize)
    root.bind("<FocusIn>", on_window_focus)
    root.bind("<FocusOut>", on_window_unfocus)
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    # Buttons
    button_frame = tk.Frame(main_frame, bg="lightgray")
    button_frame.pack(pady=10)
    
    def show_info():
        messagebox.showinfo("Info", "Window events are being monitored")
    
    tk.Button(button_frame, text="Show Info", command=show_info, bg="lightblue").pack(side="left", padx=5)
    
    root.mainloop()

# Custom event handling
def custom_event_handling():
    """Create custom event handling"""
    root = tk.Tk()
    root.title("Custom Event Handling")
    root.geometry("500x400")
    
    # Create canvas
    canvas = tk.Canvas(root, width=400, height=300, bg="white")
    canvas.pack(pady=20)
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 12))
    status_label.pack(pady=10)
    
    # Drawing variables
    drawing = False
    last_x = 0
    last_y = 0
    current_color = "black"
    current_width = 2
    
    # Custom event handlers
    def start_drawing(event):
        nonlocal drawing, last_x, last_y
        drawing = True
        last_x = event.x
        last_y = event.y
        status_label.config(text=f"Started drawing at ({event.x}, {event.y})")
    
    def draw(event):
        nonlocal last_x, last_y
        if drawing:
            canvas.create_line(last_x, last_y, event.x, event.y, fill=current_color, width=current_width)
            last_x = event.x
            last_y = event.y
            status_label.config(text=f"Drawing to ({event.x}, {event.y})")
    
    def stop_drawing(event):
        nonlocal drawing
        drawing = False
        status_label.config(text=f"Stopped drawing at ({event.x}, {event.y})")
    
    def change_color(color):
        nonlocal current_color
        current_color = color
        status_label.config(text=f"Color changed to {color}")
    
    def change_width(width):
        nonlocal current_width
        current_width = width
        status_label.config(text=f"Width changed to {width}")
    
    # Bind drawing events
    canvas.bind("<Button-1>", start_drawing)
    canvas.bind("<B1-Motion>", draw)
    canvas.bind("<ButtonRelease-1>", stop_drawing)
    
    # Control frame
    control_frame = tk.Frame(root)
    control_frame.pack(pady=10)
    
    # Color buttons
    color_frame = tk.Frame(control_frame)
    color_frame.pack(pady=5)
    
    tk.Label(color_frame, text="Colors:").pack(side="left")
    
    colors = ["black", "red", "blue", "green", "purple", "orange"]
    for color in colors:
        tk.Button(color_frame, text="", bg=color, width=3, height=1, 
                 command=lambda c=color: change_color(c)).pack(side="left", padx=2)
    
    # Width buttons
    width_frame = tk.Frame(control_frame)
    width_frame.pack(pady=5)
    
    tk.Label(width_frame, text="Width:").pack(side="left")
    
    widths = [1, 2, 3, 5, 8]
    for width in widths:
        tk.Button(width_frame, text=str(width), command=lambda w=width: change_width(w), 
                 bg="lightgray").pack(side="left", padx=2)
    
    # Control buttons
    button_frame = tk.Frame(control_frame)
    button_frame.pack(pady=5)
    
    def clear_canvas():
        canvas.delete("all")
        status_label.config(text="Canvas cleared")
    
    tk.Button(button_frame, text="Clear", command=clear_canvas, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Event binding and unbinding
def event_binding():
    """Create event binding and unbinding"""
    root = tk.Tk()
    root.title("Event Binding and Unbinding")
    root.geometry("500x400")
    
    # Create canvas
    canvas = tk.Canvas(root, width=400, height=300, bg="white")
    canvas.pack(pady=20)
    
    # Status label
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 12))
    status_label.pack(pady=10)
    
    # Event handlers
    def on_mouse_click(event):
        status_label.config(text=f"Mouse clicked at ({event.x}, {event.y})")
        canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill="red", outline="darkred", width=2)
    
    def on_mouse_motion(event):
        status_label.config(text=f"Mouse at ({event.x}, {event.y})")
    
    def on_key_press(event):
        status_label.config(text=f"Key pressed: {event.keysym}")
    
    # Binding state
    mouse_bound = False
    keyboard_bound = False
    
    def toggle_mouse_binding():
        nonlocal mouse_bound
        if mouse_bound:
            canvas.unbind("<Button-1>")
            canvas.unbind("<Motion>")
            mouse_bound = False
            status_label.config(text="Mouse events unbound")
        else:
            canvas.bind("<Button-1>", on_mouse_click)
            canvas.bind("<Motion>", on_mouse_motion)
            mouse_bound = True
            status_label.config(text="Mouse events bound")
    
    def toggle_keyboard_binding():
        nonlocal keyboard_bound
        if keyboard_bound:
            root.unbind("<KeyPress>")
            keyboard_bound = False
            status_label.config(text="Keyboard events unbound")
        else:
            root.bind("<KeyPress>", on_key_press)
            keyboard_bound = True
            status_label.config(text="Keyboard events bound")
    
    # Control frame
    control_frame = tk.Frame(root)
    control_frame.pack(pady=10)
    
    tk.Button(control_frame, text="Toggle Mouse Events", command=toggle_mouse_binding, bg="lightblue").pack(side="left", padx=5)
    tk.Button(control_frame, text="Toggle Keyboard Events", command=toggle_keyboard_binding, bg="lightgreen").pack(side="left", padx=5)
    
    # Focus the root window for keyboard events
    root.focus_set()
    
    root.mainloop()

# Event propagation
def event_propagation():
    """Create event propagation example"""
    root = tk.Tk()
    root.title("Event Propagation")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root, bg="lightgray")
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Create nested frames
    outer_frame = tk.Frame(main_frame, bg="lightblue", relief="raised", bd=2)
    outer_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    inner_frame = tk.Frame(outer_frame, bg="lightgreen", relief="raised", bd=2)
    inner_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create button
    button = tk.Button(inner_frame, text="Click Me", bg="lightcoral", font=("Arial", 12, "bold"))
    button.pack(expand=True)
    
    # Status label
    status_label = tk.Label(main_frame, text="Status: Ready", font=("Arial", 12), bg="lightgray")
    status_label.pack(pady=10)
    
    # Event handlers
    def on_outer_click(event):
        status_label.config(text="Outer frame clicked")
    
    def on_inner_click(event):
        status_label.config(text="Inner frame clicked")
    
    def on_button_click(event):
        status_label.config(text="Button clicked")
    
    def on_button_click_command():
        status_label.config(text="Button command executed")
    
    # Bind events
    outer_frame.bind("<Button-1>", on_outer_click)
    inner_frame.bind("<Button-1>", on_inner_click)
    button.bind("<Button-1>", on_button_click)
    button.config(command=on_button_click_command)
    
    # Control frame
    control_frame = tk.Frame(main_frame, bg="lightgray")
    control_frame.pack(pady=10)
    
    def clear_status():
        status_label.config(text="Status: Ready")
    
    tk.Button(control_frame, text="Clear Status", command=clear_status, bg="lightyellow").pack(side="left", padx=5)
    
    root.mainloop()

# Complete event example
def complete_event_example():
    """Complete example showing all event features"""
    root = tk.Tk()
    root.title("Complete Event Example")
    root.geometry("700x600")
    
    # Header
    header = tk.Label(root, text="Complete Event Example", 
                     font=("Arial", 16, "bold"), bg="darkblue", fg="white")
    header.pack(fill="x", padx=10, pady=10)
    
    # Main content frame
    content_frame = tk.Frame(root, bg="lightgray")
    content_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Left column
    left_frame = tk.Frame(content_frame, bg="lightblue", relief="raised", bd=2)
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_frame, text="Mouse Events", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
    
    # Mouse event canvas
    mouse_canvas = tk.Canvas(left_frame, width=250, height=150, bg="white")
    mouse_canvas.pack(pady=10)
    
    def on_mouse_click(event):
        mouse_canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill="red", outline="darkred", width=2)
    
    mouse_canvas.bind("<Button-1>", on_mouse_click)
    
    # Right column
    right_frame = tk.Frame(content_frame, bg="lightgreen", relief="raised", bd=2)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_frame, text="Keyboard Events", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=10)
    
    # Keyboard event text widget
    keyboard_text = tk.Text(right_frame, width=30, height=8, wrap="word")
    keyboard_text.pack(pady=10)
    
    def on_key_press(event):
        keyboard_text.insert(tk.END, f"Key: {event.keysym}\n")
        keyboard_text.see(tk.END)
    
    keyboard_text.bind("<KeyPress>", on_key_press)
    keyboard_text.focus_set()
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Event Examples")
    print("=" * 30)
    print("1. Basic Mouse Events")
    print("2. Keyboard Events")
    print("3. Window Events")
    print("4. Custom Event Handling")
    print("5. Event Binding")
    print("6. Event Propagation")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_mouse_events()
    elif choice == "2":
        keyboard_events()
    elif choice == "3":
        window_events()
    elif choice == "4":
        custom_event_handling()
    elif choice == "5":
        event_binding()
    elif choice == "6":
        event_propagation()
    elif choice == "7":
        complete_event_example()
    else:
        print("Invalid choice. Running basic mouse events...")
        basic_mouse_events()

"""
Event Types:
------------
- Mouse Events: <Button-1>, <B1-Motion>, <ButtonRelease-1>, <Motion>
- Keyboard Events: <KeyPress>, <KeyRelease>, <Key>
- Window Events: <Configure>, <FocusIn>, <FocusOut>
- Custom Events: User-defined events

Event Properties:
-----------------
- event.x, event.y: Mouse coordinates
- event.keysym: Key symbol
- event.char: Character representation
- event.width, event.height: Widget dimensions
- event.widget: Widget that triggered the event

Event Binding:
--------------
- bind(): Bind event to widget
- unbind(): Unbind event from widget
- bind_all(): Bind event to all widgets
- unbind_all(): Unbind event from all widgets

Best Practices:
--------------
- Use appropriate event types for different interactions
- Handle events efficiently to avoid performance issues
- Provide visual feedback for user actions
- Consider accessibility and keyboard navigation
- Test event handling with different input devices
- Use event propagation carefully

Extra Tips:
-----------
- Use focus_set() to enable keyboard events
- Use return "break" to stop event propagation
- Use after() for delayed event handling
- Use update_idletasks() to process pending events
- Consider using ttk widgets for modern styling
- Use event.x_root and event.y_root for screen coordinates
"""