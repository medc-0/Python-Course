"""
15-tkinter-canvas.py

Tkinter Canvas - Drawing and graphics widget

Overview:
---------
Canvas is a powerful widget for drawing graphics, shapes, and images. 
It's perfect for creating custom graphics, games, data visualizations, 
and interactive drawing applications.

Key Features:
- Draw shapes, lines, and text
- Display images and graphics
- Handle mouse and keyboard events
- Create animations and games
- Custom drawing and painting

Common Use Cases:
- Drawing applications and paint programs
- Games and interactive graphics
- Data visualization and charts
- Custom widgets and controls
- Educational tools and simulations

Tips:
- Use appropriate drawing methods for different shapes
- Handle mouse events for interactive drawing
- Consider performance for complex graphics
- Use tags for grouping and manipulating objects
- Test with different screen sizes and resolutions
"""

import tkinter as tk
from tkinter import messagebox
import math

# Basic canvas drawing
def basic_canvas():
    """Create basic canvas with shapes"""
    root = tk.Tk()
    root.title("Basic Canvas")
    root.geometry("500x400")
    
    # Create canvas
    canvas = tk.Canvas(root, width=400, height=300, bg="white")
    canvas.pack(pady=20)
    
    # Draw shapes
    canvas.create_rectangle(50, 50, 150, 100, fill="lightblue", outline="blue", width=2)
    canvas.create_oval(200, 50, 300, 100, fill="lightgreen", outline="green", width=2)
    canvas.create_line(50, 150, 300, 150, fill="red", width=3)
    canvas.create_text(175, 200, text="Canvas Drawing", font=("Arial", 14, "bold"), fill="purple")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def clear_canvas():
        canvas.delete("all")
    
    def add_shape():
        canvas.create_rectangle(100, 100, 200, 150, fill="yellow", outline="orange", width=2)
    
    tk.Button(button_frame, text="Add Shape", command=add_shape, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", command=clear_canvas, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Interactive canvas
def interactive_canvas():
    """Create interactive canvas with mouse events"""
    root = tk.Tk()
    root.title("Interactive Canvas")
    root.geometry("600x500")
    
    # Create canvas
    canvas = tk.Canvas(root, width=500, height=400, bg="white")
    canvas.pack(pady=20)
    
    # Drawing variables
    drawing = False
    last_x = 0
    last_y = 0
    
    def start_drawing(event):
        nonlocal drawing, last_x, last_y
        drawing = True
        last_x = event.x
        last_y = event.y
    
    def draw(event):
        nonlocal last_x, last_y
        if drawing:
            canvas.create_line(last_x, last_y, event.x, event.y, fill="black", width=2)
            last_x = event.x
            last_y = event.y
    
    def stop_drawing(event):
        nonlocal drawing
        drawing = False
    
    # Bind mouse events
    canvas.bind("<Button-1>", start_drawing)
    canvas.bind("<B1-Motion>", draw)
    canvas.bind("<ButtonRelease-1>", stop_drawing)
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def clear_canvas():
        canvas.delete("all")
    
    def change_color(color):
        canvas.config(bg=color)
    
    tk.Button(button_frame, text="Clear", command=clear_canvas, bg="lightcoral").pack(side="left", padx=5)
    tk.Button(button_frame, text="White", command=lambda: change_color("white"), bg="white").pack(side="left", padx=5)
    tk.Button(button_frame, text="Light Blue", command=lambda: change_color("lightblue"), bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Light Green", command=lambda: change_color("lightgreen"), bg="lightgreen").pack(side="left", padx=5)
    
    root.mainloop()

# Canvas with shapes
def canvas_shapes():
    """Create canvas with various shapes"""
    root = tk.Tk()
    root.title("Canvas Shapes")
    root.geometry("600x500")
    
    # Create canvas
    canvas = tk.Canvas(root, width=500, height=400, bg="white")
    canvas.pack(pady=20)
    
    # Draw various shapes
    def draw_shapes():
        # Rectangle
        canvas.create_rectangle(50, 50, 150, 100, fill="lightblue", outline="blue", width=2)
        
        # Oval
        canvas.create_oval(200, 50, 300, 100, fill="lightgreen", outline="green", width=2)
        
        # Line
        canvas.create_line(50, 150, 300, 150, fill="red", width=3)
        
        # Polygon
        canvas.create_polygon(50, 200, 100, 250, 150, 200, fill="lightyellow", outline="orange", width=2)
        
        # Arc
        canvas.create_arc(200, 200, 300, 300, start=0, extent=180, fill="lightcoral", outline="red", width=2)
        
        # Text
        canvas.create_text(175, 350, text="Various Shapes", font=("Arial", 14, "bold"), fill="purple")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def clear_canvas():
        canvas.delete("all")
    
    def add_random_shape():
        import random
        x = random.randint(50, 400)
        y = random.randint(50, 350)
        color = random.choice(["lightblue", "lightgreen", "lightyellow", "lightcoral", "lightpink"])
        canvas.create_oval(x-25, y-25, x+25, y+25, fill=color, outline="black", width=2)
    
    tk.Button(button_frame, text="Draw Shapes", command=draw_shapes, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Add Random", command=add_random_shape, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", command=clear_canvas, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Canvas with animations
def canvas_animations():
    """Create canvas with animations"""
    root = tk.Tk()
    root.title("Canvas Animations")
    root.geometry("600x500")
    
    # Create canvas
    canvas = tk.Canvas(root, width=500, height=400, bg="white")
    canvas.pack(pady=20)
    
    # Animation variables
    ball_x = 250
    ball_y = 200
    ball_dx = 2
    ball_dy = 2
    ball_id = None
    
    def create_ball():
        nonlocal ball_id
        ball_id = canvas.create_oval(ball_x-10, ball_y-10, ball_x+10, ball_y+10, fill="red", outline="darkred", width=2)
    
    def animate_ball():
        nonlocal ball_x, ball_y, ball_dx, ball_dy
        
        # Move ball
        ball_x += ball_dx
        ball_y += ball_dy
        
        # Bounce off walls
        if ball_x <= 10 or ball_x >= 490:
            ball_dx = -ball_dx
        if ball_y <= 10 or ball_y >= 390:
            ball_dy = -ball_dy
        
        # Update ball position
        canvas.coords(ball_id, ball_x-10, ball_y-10, ball_x+10, ball_y+10)
        
        # Continue animation
        root.after(20, animate_ball)
    
    def start_animation():
        create_ball()
        animate_ball()
    
    def stop_animation():
        if ball_id:
            canvas.delete(ball_id)
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Start Animation", command=start_animation, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Stop Animation", command=stop_animation, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Canvas with mouse drawing
def canvas_mouse_drawing():
    """Create canvas with mouse drawing"""
    root = tk.Tk()
    root.title("Mouse Drawing Canvas")
    root.geometry("600x500")
    
    # Create canvas
    canvas = tk.Canvas(root, width=500, height=400, bg="white")
    canvas.pack(pady=20)
    
    # Drawing variables
    drawing = False
    last_x = 0
    last_y = 0
    current_color = "black"
    current_width = 2
    
    def start_drawing(event):
        nonlocal drawing, last_x, last_y
        drawing = True
        last_x = event.x
        last_y = event.y
    
    def draw(event):
        nonlocal last_x, last_y
        if drawing:
            canvas.create_line(last_x, last_y, event.x, event.y, fill=current_color, width=current_width)
            last_x = event.x
            last_y = event.y
    
    def stop_drawing(event):
        nonlocal drawing
        drawing = False
    
    # Bind mouse events
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
                 command=lambda c=color: set_color(c)).pack(side="left", padx=2)
    
    def set_color(color):
        nonlocal current_color
        current_color = color
    
    # Width control
    width_frame = tk.Frame(control_frame)
    width_frame.pack(pady=5)
    
    tk.Label(width_frame, text="Width:").pack(side="left")
    
    widths = [1, 2, 3, 5, 8]
    for width in widths:
        tk.Button(width_frame, text=str(width), command=lambda w=width: set_width(w), 
                 bg="lightgray").pack(side="left", padx=2)
    
    def set_width(width):
        nonlocal current_width
        current_width = width
    
    # Control buttons
    button_frame = tk.Frame(control_frame)
    button_frame.pack(pady=5)
    
    def clear_canvas():
        canvas.delete("all")
    
    def save_drawing():
        messagebox.showinfo("Save", "Drawing saved (simulated)")
    
    tk.Button(button_frame, text="Clear", command=clear_canvas, bg="lightcoral").pack(side="left", padx=5)
    tk.Button(button_frame, text="Save", command=save_drawing, bg="lightblue").pack(side="left", padx=5)
    
    root.mainloop()

# Canvas with data visualization
def canvas_data_visualization():
    """Create canvas with data visualization"""
    root = tk.Tk()
    root.title("Data Visualization Canvas")
    root.geometry("700x600")
    
    # Create canvas
    canvas = tk.Canvas(root, width=600, height=500, bg="white")
    canvas.pack(pady=20)
    
    # Sample data
    data = [20, 35, 30, 45, 25, 40, 50, 35, 20, 30]
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    
    def draw_bar_chart():
        canvas.delete("all")
        
        # Chart dimensions
        chart_width = 500
        chart_height = 400
        chart_x = 50
        chart_y = 50
        
        # Draw axes
        canvas.create_line(chart_x, chart_y + chart_height, chart_x + chart_width, chart_y + chart_height, fill="black", width=2)
        canvas.create_line(chart_x, chart_y, chart_x, chart_y + chart_height, fill="black", width=2)
        
        # Draw bars
        bar_width = chart_width // len(data)
        max_value = max(data)
        
        for i, value in enumerate(data):
            bar_height = (value / max_value) * chart_height
            x1 = chart_x + i * bar_width
            y1 = chart_y + chart_height - bar_height
            x2 = x1 + bar_width - 5
            y2 = chart_y + chart_height
            
            # Draw bar
            canvas.create_rectangle(x1, y1, x2, y2, fill="lightblue", outline="blue", width=1)
            
            # Draw label
            canvas.create_text(x1 + bar_width//2, chart_y + chart_height + 20, text=labels[i], font=("Arial", 10))
            
            # Draw value
            canvas.create_text(x1 + bar_width//2, y1 - 10, text=str(value), font=("Arial", 8))
    
    def draw_line_chart():
        canvas.delete("all")
        
        # Chart dimensions
        chart_width = 500
        chart_height = 400
        chart_x = 50
        chart_y = 50
        
        # Draw axes
        canvas.create_line(chart_x, chart_y + chart_height, chart_x + chart_width, chart_y + chart_height, fill="black", width=2)
        canvas.create_line(chart_x, chart_y, chart_x, chart_y + chart_height, fill="black", width=2)
        
        # Draw line
        max_value = max(data)
        points = []
        
        for i, value in enumerate(data):
            x = chart_x + (i / (len(data) - 1)) * chart_width
            y = chart_y + chart_height - (value / max_value) * chart_height
            points.extend([x, y])
        
        canvas.create_line(points, fill="red", width=3)
        
        # Draw points
        for i, value in enumerate(data):
            x = chart_x + (i / (len(data) - 1)) * chart_width
            y = chart_y + chart_height - (value / max_value) * chart_height
            canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", outline="darkred", width=2)
    
    def draw_pie_chart():
        canvas.delete("all")
        
        # Chart dimensions
        chart_width = 400
        chart_height = 400
        chart_x = 100
        chart_y = 50
        
        # Calculate total
        total = sum(data)
        
        # Draw pie slices
        start_angle = 0
        colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral", "lightpink", 
                 "lightgray", "lightcyan", "lightsteelblue", "lightseagreen", "lightpink"]
        
        for i, value in enumerate(data):
            extent = (value / total) * 360
            canvas.create_arc(chart_x, chart_y, chart_x + chart_width, chart_y + chart_height,
                           start=start_angle, extent=extent, fill=colors[i % len(colors)], outline="black", width=2)
            start_angle += extent
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Bar Chart", command=draw_bar_chart, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Line Chart", command=draw_line_chart, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Pie Chart", command=draw_pie_chart, bg="lightyellow").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", command=lambda: canvas.delete("all"), bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Complete canvas example
def complete_canvas_example():
    """Complete example showing all canvas features"""
    root = tk.Tk()
    root.title("Complete Canvas Example")
    root.geometry("800x700")
    
    # Header
    header = tk.Label(root, text="Complete Canvas Example", 
                     font=("Arial", 16, "bold"), bg="darkblue", fg="white")
    header.pack(fill="x", padx=10, pady=10)
    
    # Main content frame
    content_frame = tk.Frame(root, bg="lightgray")
    content_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Canvas frame
    canvas_frame = tk.Frame(content_frame, bg="white", relief="sunken", bd=2)
    canvas_frame.pack(expand=True, fill="both", pady=(0, 10))
    
    # Create canvas
    canvas = tk.Canvas(canvas_frame, width=700, height=500, bg="white")
    canvas.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Draw sample content
    def draw_sample_content():
        canvas.delete("all")
        
        # Draw various shapes
        canvas.create_rectangle(50, 50, 150, 100, fill="lightblue", outline="blue", width=2)
        canvas.create_oval(200, 50, 300, 100, fill="lightgreen", outline="green", width=2)
        canvas.create_line(50, 150, 300, 150, fill="red", width=3)
        canvas.create_text(175, 200, text="Canvas Drawing", font=("Arial", 14, "bold"), fill="purple")
        
        # Draw grid
        for i in range(0, 700, 50):
            canvas.create_line(i, 0, i, 500, fill="lightgray", width=1)
        for i in range(0, 500, 50):
            canvas.create_line(0, i, 700, i, fill="lightgray", width=1)
    
    # Control frame
    control_frame = tk.Frame(content_frame)
    control_frame.pack(fill="x")
    
    # Buttons
    button_frame = tk.Frame(control_frame)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Draw Content", command=draw_sample_content, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", command=lambda: canvas.delete("all"), bg="lightcoral").pack(side="left", padx=5)
    tk.Button(button_frame, text="Add Shape", command=lambda: canvas.create_oval(100, 100, 200, 150, fill="yellow", outline="orange", width=2), bg="lightgreen").pack(side="left", padx=5)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Canvas Examples")
    print("=" * 30)
    print("1. Basic Canvas")
    print("2. Interactive Canvas")
    print("3. Canvas Shapes")
    print("4. Canvas Animations")
    print("5. Mouse Drawing")
    print("6. Data Visualization")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_canvas()
    elif choice == "2":
        interactive_canvas()
    elif choice == "3":
        canvas_shapes()
    elif choice == "4":
        canvas_animations()
    elif choice == "5":
        canvas_mouse_drawing()
    elif choice == "6":
        canvas_data_visualization()
    elif choice == "7":
        complete_canvas_example()
    else:
        print("Invalid choice. Running basic canvas...")
        basic_canvas()

"""
Canvas Drawing Methods:
-----------------------
- create_rectangle(): Draw rectangle
- create_oval(): Draw oval/circle
- create_line(): Draw line
- create_polygon(): Draw polygon
- create_arc(): Draw arc
- create_text(): Draw text
- create_image(): Display image

Canvas Properties:
------------------
- width: Canvas width
- height: Canvas height
- bg: Background color
- scrollregion: Scrollable region
- xscrollcommand: Horizontal scrollbar
- yscrollcommand: Vertical scrollbar

Canvas Events:
--------------
- <Button-1>: Left mouse click
- <B1-Motion>: Left mouse drag
- <ButtonRelease-1>: Left mouse release
- <Motion>: Mouse movement
- <KeyPress>: Key press
- <KeyRelease>: Key release

Best Practices:
--------------
- Use appropriate drawing methods for different shapes
- Handle mouse events for interactive drawing
- Consider performance for complex graphics
- Use tags for grouping and manipulating objects
- Test with different screen sizes and resolutions
- Use error handling for drawing operations

Extra Tips:
-----------
- Use coords() to move objects
- Use delete() to remove objects
- Use tags for grouping objects
- Use find_all() to find objects
- Use bbox() to get object bounds
- Consider using PIL for advanced image processing
"""