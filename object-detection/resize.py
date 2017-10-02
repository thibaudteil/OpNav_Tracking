'''
    resize.py
    Resize images to 300x200 while keeping aspect ratio same
'''

# Some imports
import os
from PIL import Image

# Original image directory
root = ".."
extension = "jpg"
original_data_path = os.path.join(root, "data", "NASA", "LRO")

# New directory for resized images
new_dir = "images"

# New size for images
width = 300
height = 200

# Get all original images
original_files = os.listdir(original_data_path)

# Convert all original images
for File in original_files:

    # Make sure the file is an image
    if File[-3:] == extension:
        # Conversion and Saving
        im = Image.open(os.path.join(original_data_path, File))
        im = im.resize((width, height), Image.ANTIALIAS)
        path_to_save = os.path.join(new_dir, File)
        im.save(path_to_save)
