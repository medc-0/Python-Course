"""
14-tkinter-images.py

Tkinter Images - Display and manipulate images in your GUI

Overview:
---------
Tkinter provides support for displaying images in your GUI applications. 
You can show images in labels, buttons, and other widgets, and even 
create simple image viewers and editors.

Key Features:
- Display images in labels and buttons
- Support for various image formats
- Image resizing and manipulation
- Image viewers and galleries
- Custom image widgets

Common Use Cases:
- Image viewers and galleries
- Photo albums and slideshows
- Icons and graphics in buttons
- Data visualization
- Game graphics and sprites

Tips:
- Use appropriate image formats (PNG, GIF, JPEG)
- Consider image size and memory usage
- Provide fallbacks for missing images
- Use image resizing for different screen sizes
- Test with various image formats
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import os

# Basic image display
def basic_image_display():
    """Create basic image display"""
    root = tk.Tk()
    root.title("Basic Image Display")
    root.geometry("500x400")
    
    # Create image label
    image_label = tk.Label(root, text="No image loaded", bg="lightgray", width=50, height=20)
    image_label.pack(pady=20)
    
    # Load image function
    def load_image():
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                image = image.resize((400, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                image_label.config(image=photo, text="")
                image_label.image = photo  # Keep a reference
                
                messagebox.showinfo("Success", f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Load Image", command=load_image, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear Image", command=lambda: image_label.config(image="", text="No image loaded"), 
              bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Image viewer with controls
def image_viewer():
    """Create image viewer with controls"""
    root = tk.Tk()
    root.title("Image Viewer")
    root.geometry("600x500")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Image display frame
    image_frame = tk.Frame(main_frame, bg="lightgray", relief="sunken", bd=2)
    image_frame.pack(expand=True, fill="both", pady=(0, 10))
    
    # Image label
    image_label = tk.Label(image_frame, text="No image loaded", bg="lightgray")
    image_label.pack(expand=True)
    
    # Control frame
    control_frame = tk.Frame(main_frame)
    control_frame.pack(fill="x")
    
    # Image info
    info_label = tk.Label(control_frame, text="No image loaded", font=("Arial", 10))
    info_label.pack(pady=5)
    
    # Buttons
    button_frame = tk.Frame(control_frame)
    button_frame.pack(pady=5)
    
    def load_image():
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Load image
                image = Image.open(file_path)
                original_size = image.size
                
                # Resize if too large
                if image.width > 500 or image.height > 400:
                    image.thumbnail((500, 400), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                image_label.config(image=photo, text="")
                image_label.image = photo
                
                # Update info
                info_label.config(text=f"Image: {os.path.basename(file_path)} | Size: {original_size[0]}x{original_size[1]}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")
    
    def clear_image():
        image_label.config(image="", text="No image loaded")
        image_label.image = None
        info_label.config(text="No image loaded")
    
    tk.Button(button_frame, text="Load Image", command=load_image, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear Image", command=clear_image, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Image gallery
def image_gallery():
    """Create image gallery"""
    root = tk.Tk()
    root.title("Image Gallery")
    root.geometry("700x600")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Gallery frame
    gallery_frame = tk.Frame(main_frame, bg="lightgray", relief="sunken", bd=2)
    gallery_frame.pack(expand=True, fill="both", pady=(0, 10))
    
    # Gallery label
    gallery_label = tk.Label(gallery_frame, text="No images loaded", bg="lightgray")
    gallery_label.pack(expand=True)
    
    # Control frame
    control_frame = tk.Frame(main_frame)
    control_frame.pack(fill="x")
    
    # Image info
    info_label = tk.Label(control_frame, text="No images loaded", font=("Arial", 10))
    info_label.pack(pady=5)
    
    # Buttons
    button_frame = tk.Frame(control_frame)
    button_frame.pack(pady=5)
    
    # Gallery variables
    current_image = 0
    images = []
    
    def load_images():
        nonlocal current_image, images
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")]
        )
        if file_paths:
            try:
                images = []
                for file_path in file_paths:
                    image = Image.open(file_path)
                    if image.width > 300 or image.height > 300:
                        image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    images.append((photo, os.path.basename(file_path)))
                
                current_image = 0
                display_current_image()
                info_label.config(text=f"Image {current_image + 1} of {len(images)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load images: {e}")
    
    def display_current_image():
        if images:
            photo, filename = images[current_image]
            gallery_label.config(image=photo, text="")
            gallery_label.image = photo
            info_label.config(text=f"Image {current_image + 1} of {len(images)}: {filename}")
    
    def next_image():
        nonlocal current_image
        if images:
            current_image = (current_image + 1) % len(images)
            display_current_image()
    
    def previous_image():
        nonlocal current_image
        if images:
            current_image = (current_image - 1) % len(images)
            display_current_image()
    
    def clear_images():
        nonlocal current_image, images
        images = []
        current_image = 0
        gallery_label.config(image="", text="No images loaded")
        gallery_label.image = None
        info_label.config(text="No images loaded")
    
    tk.Button(button_frame, text="Load Images", command=load_images, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Previous", command=previous_image, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Next", command=next_image, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", command=clear_images, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Image with buttons
def image_buttons():
    """Create buttons with images"""
    root = tk.Tk()
    root.title("Image Buttons")
    root.geometry("500x400")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Image buttons frame
    buttons_frame = tk.Frame(main_frame)
    buttons_frame.pack(expand=True)
    
    # Create image buttons
    def create_image_button(text, bg_color):
        # Create a simple colored button with text
        button = tk.Button(buttons_frame, text=text, bg=bg_color, font=("Arial", 12, "bold"),
                          width=15, height=3)
        return button
    
    # Create buttons
    btn1 = create_image_button("Button 1", "lightblue")
    btn1.pack(pady=10)
    
    btn2 = create_image_button("Button 2", "lightgreen")
    btn2.pack(pady=10)
    
    btn3 = create_image_button("Button 3", "lightcoral")
    btn3.pack(pady=10)
    
    # Status label
    status_label = tk.Label(main_frame, text="Click a button", font=("Arial", 12))
    status_label.pack(pady=10)
    
    # Button click handlers
    def on_button_click(button_text):
        status_label.config(text=f"Clicked: {button_text}")
    
    btn1.config(command=lambda: on_button_click("Button 1"))
    btn2.config(command=lambda: on_button_click("Button 2"))
    btn3.config(command=lambda: on_button_click("Button 3"))
    
    root.mainloop()

# Image resizing
def image_resizing():
    """Create image resizing functionality"""
    root = tk.Tk()
    root.title("Image Resizing")
    root.geometry("600x500")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Image display frame
    image_frame = tk.Frame(main_frame, bg="lightgray", relief="sunken", bd=2)
    image_frame.pack(expand=True, fill="both", pady=(0, 10))
    
    # Image label
    image_label = tk.Label(image_frame, text="No image loaded", bg="lightgray")
    image_label.pack(expand=True)
    
    # Control frame
    control_frame = tk.Frame(main_frame)
    control_frame.pack(fill="x")
    
    # Image info
    info_label = tk.Label(control_frame, text="No image loaded", font=("Arial", 10))
    info_label.pack(pady=5)
    
    # Size control frame
    size_frame = tk.Frame(control_frame)
    size_frame.pack(pady=5)
    
    tk.Label(size_frame, text="Size:").pack(side="left")
    size_var = tk.StringVar()
    size_var.set("300x300")
    size_entry = tk.Entry(size_frame, textvariable=size_var, width=10)
    size_entry.pack(side="left", padx=5)
    
    # Buttons
    button_frame = tk.Frame(control_frame)
    button_frame.pack(pady=5)
    
    # Image variables
    current_image = None
    
    def load_image():
        nonlocal current_image
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            try:
                current_image = Image.open(file_path)
                resize_image()
                info_label.config(text=f"Image: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")
    
    def resize_image():
        if current_image:
            try:
                size_text = size_var.get()
                width, height = map(int, size_text.split('x'))
                
                resized_image = current_image.resize((width, height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(resized_image)
                
                image_label.config(image=photo, text="")
                image_label.image = photo
                
            except Exception as e:
                messagebox.showerror("Error", f"Invalid size format: {e}")
    
    def clear_image():
        nonlocal current_image
        current_image = None
        image_label.config(image="", text="No image loaded")
        image_label.image = None
        info_label.config(text="No image loaded")
    
    tk.Button(button_frame, text="Load Image", command=load_image, bg="lightblue").pack(side="left", padx=5)
    tk.Button(button_frame, text="Resize", command=resize_image, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Clear", command=clear_image, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Image with effects
def image_effects():
    """Create image effects"""
    root = tk.Tk()
    root.title("Image Effects")
    root.geometry("600x500")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Image display frame
    image_frame = tk.Frame(main_frame, bg="lightgray", relief="sunken", bd=2)
    image_frame.pack(expand=True, fill="both", pady=(0, 10))
    
    # Image label
    image_label = tk.Label(image_frame, text="No image loaded", bg="lightgray")
    image_label.pack(expand=True)
    
    # Control frame
    control_frame = tk.Frame(main_frame)
    control_frame.pack(fill="x")
    
    # Image info
    info_label = tk.Label(control_frame, text="No image loaded", font=("Arial", 10))
    info_label.pack(pady=5)
    
    # Buttons
    button_frame = tk.Frame(control_frame)
    button_frame.pack(pady=5)
    
    # Image variables
    current_image = None
    original_image = None
    
    def load_image():
        nonlocal current_image, original_image
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            try:
                current_image = Image.open(file_path)
                original_image = current_image.copy()
                display_image()
                info_label.config(text=f"Image: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")
    
    def display_image():
        if current_image:
            # Resize if too large
            if current_image.width > 400 or current_image.height > 300:
                current_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(current_image)
            image_label.config(image=photo, text="")
            image_label.image = photo
    
    def apply_effect(effect):
        if current_image:
            try:
                if effect == "grayscale":
                    current_image = current_image.convert("L").convert("RGB")
                elif effect == "blur":
                    current_image = current_image.filter(Image.Filter.BLUR)
                elif effect == "sharpen":
                    current_image = current_image.filter(Image.Filter.SHARPEN)
                elif effect == "emboss":
                    current_image = current_image.filter(Image.Filter.EMBOSS)
                elif effect == "edge_enhance":
                    current_image = current_image.filter(Image.Filter.EDGE_ENHANCE)
                
                display_image()
            except Exception as e:
                messagebox.showerror("Error", f"Could not apply effect: {e}")
    
    def reset_image():
        nonlocal current_image
        if original_image:
            current_image = original_image.copy()
            display_image()
    
    def clear_image():
        nonlocal current_image, original_image
        current_image = None
        original_image = None
        image_label.config(image="", text="No image loaded")
        image_label.image = None
        info_label.config(text="No image loaded")
    
    # Effect buttons
    effect_frame = tk.Frame(control_frame)
    effect_frame.pack(pady=5)
    
    tk.Button(effect_frame, text="Grayscale", command=lambda: apply_effect("grayscale"), bg="lightgray").pack(side="left", padx=2)
    tk.Button(effect_frame, text="Blur", command=lambda: apply_effect("blur"), bg="lightblue").pack(side="left", padx=2)
    tk.Button(effect_frame, text="Sharpen", command=lambda: apply_effect("sharpen"), bg="lightgreen").pack(side="left", padx=2)
    tk.Button(effect_frame, text="Emboss", command=lambda: apply_effect("emboss"), bg="lightyellow").pack(side="left", padx=2)
    tk.Button(effect_frame, text="Edge Enhance", command=lambda: apply_effect("edge_enhance"), bg="lightcoral").pack(side="left", padx=2)
    
    # Control buttons
    control_button_frame = tk.Frame(control_frame)
    control_button_frame.pack(pady=5)
    
    tk.Button(control_button_frame, text="Load Image", command=load_image, bg="lightblue").pack(side="left", padx=5)
    tk.Button(control_button_frame, text="Reset", command=reset_image, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(control_button_frame, text="Clear", command=clear_image, bg="lightcoral").pack(side="left", padx=5)
    
    root.mainloop()

# Complete image example
def complete_image_example():
    """Complete example showing all image features"""
    root = tk.Tk()
    root.title("Complete Image Example")
    root.geometry("800x600")
    
    # Header
    header = tk.Label(root, text="Complete Image Example", 
                     font=("Arial", 16, "bold"), bg="darkblue", fg="white")
    header.pack(fill="x", padx=10, pady=10)
    
    # Main content frame
    content_frame = tk.Frame(root, bg="lightgray")
    content_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    # Left column
    left_frame = tk.Frame(content_frame, bg="lightblue", relief="raised", bd=2)
    left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(left_frame, text="Image Operations", font=("Arial", 12, "bold"), bg="lightblue").pack(pady=10)
    
    tk.Button(left_frame, text="Load Image", command=lambda: filedialog.askopenfilename(), 
              bg="white").pack(pady=5, fill="x", padx=10)
    tk.Button(left_frame, text="Save Image", command=lambda: filedialog.asksaveasfilename(), 
              bg="white").pack(pady=5, fill="x", padx=10)
    tk.Button(left_frame, text="Resize Image", command=lambda: messagebox.showinfo("Info", "Resize functionality"), 
              bg="white").pack(pady=5, fill="x", padx=10)
    
    # Right column
    right_frame = tk.Frame(content_frame, bg="lightgreen", relief="raised", bd=2)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    tk.Label(right_frame, text="Image Effects", font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=10)
    
    tk.Button(right_frame, text="Grayscale", command=lambda: messagebox.showinfo("Info", "Grayscale effect"), 
              bg="white").pack(pady=5, fill="x", padx=10)
    tk.Button(right_frame, text="Blur", command=lambda: messagebox.showinfo("Info", "Blur effect"), 
              bg="white").pack(pady=5, fill="x", padx=10)
    tk.Button(right_frame, text="Sharpen", command=lambda: messagebox.showinfo("Info", "Sharpen effect"), 
              bg="white").pack(pady=5, fill="x", padx=10)
    
    # Status bar
    status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
    status.pack(side="bottom", fill="x")
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Tkinter Image Examples")
    print("=" * 30)
    print("1. Basic Image Display")
    print("2. Image Viewer")
    print("3. Image Gallery")
    print("4. Image Buttons")
    print("5. Image Resizing")
    print("6. Image Effects")
    print("7. Complete Example")
    print("=" * 30)
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == "1":
        basic_image_display()
    elif choice == "2":
        image_viewer()
    elif choice == "3":
        image_gallery()
    elif choice == "4":
        image_buttons()
    elif choice == "5":
        image_resizing()
    elif choice == "6":
        image_effects()
    elif choice == "7":
        complete_image_example()
    else:
        print("Invalid choice. Running basic image display...")
        basic_image_display()

"""
Image Formats:
--------------
- PNG: Best for graphics with transparency
- JPEG: Best for photographs
- GIF: Best for simple graphics and animations
- BMP: Uncompressed bitmap format

Image Operations:
----------------
- open(): Open image file
- resize(): Resize image
- thumbnail(): Create thumbnail
- convert(): Convert image format
- filter(): Apply image filters
- save(): Save image to file

Common Filters:
---------------
- BLUR: Blur effect
- SHARPEN: Sharpen effect
- EMBOSS: Emboss effect
- EDGE_ENHANCE: Edge enhancement
- SMOOTH: Smooth effect

Best Practices:
--------------
- Use appropriate image formats
- Consider image size and memory usage
- Provide fallbacks for missing images
- Use image resizing for different screen sizes
- Test with various image formats
- Use error handling for image operations

Extra Tips:
-----------
- Use PIL (Pillow) for advanced image processing
- Use thumbnail() for efficient resizing
- Use convert() for format conversion
- Use filter() for image effects
- Consider using ttk widgets for modern styling
- Use image caching for better performance
"""
