"""
10-image-automation.py

Beginner's guide to image automation with Pillow.

Overview:
---------
Pillow is a popular Python library for image processing. You can create, open, edit, resize, crop, and save images automatically.
This is useful for batch editing, resizing photos, converting formats, or creating graphics.

Main Features:
--------------
- Create new images
- Open and edit existing images
- Resize, crop, rotate, and flip images
- Convert between formats (PNG, JPEG, etc.)
- Save images to disk

Examples:
---------
"""

from PIL import Image

# 1. Create a new image (red square)
img = Image.new("RGB", (100, 100), color="red")
img.save("red.png")

# 2. Open and resize an image
img = Image.open("red.png")
img_resized = img.resize((50, 50))
img_resized.save("red_small.png")

# 3. Crop an image
img_cropped = img.crop((10, 10, 60, 60))  # (left, top, right, bottom)
img_cropped.save("red_cropped.png")

# 4. Rotate an image
img_rotated = img.rotate(45)
img_rotated.save("red_rotated.png")

# 5. Convert image format
img.save("red.jpg", "JPEG")

print("Image automation examples done.")

# Clean up
import os
os.remove("red.png")
os.remove("red_small.png")
os.remove("red_cropped.png")
os.remove("red_rotated.png")
os.remove("red.jpg")

"""
Tips:
-----
- Install Pillow: pip install pillow
- Pillow supports many image formats and operations.
- Use for batch editing, resizing, or format conversion.
- Official docs: https://pillow.readthedocs.io/en/stable/
"""
