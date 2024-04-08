import os
from PIL import Image
import random


NUM_IMAGES = 5
# Directory containing the images
dir1 = '/Users/ryanslocum/Documents/class/CSCI5922/project/phenobench/submissions/5265'
dir2 = '/Users/ryanslocum/Documents/class/CSCI5922/project/phenobench/data/PhenoBench/test/images'
# Get a list of image files in the directory
mask_files = [os.path.join(dir1, file) for file in os.listdir(dir1) if file.endswith('.png')]
random.shuffle(mask_files)
image_files = [os.path.join(dir2, file) for file in os.listdir(dir2) if file.endswith('.png')]
random.shuffle(image_files)

# Interleave the two lists
image_files = [val for pair in zip(mask_files, image_files) for val in pair]

# Create a blank canvas for the collage
collage = Image.new('RGB', (1024*NUM_IMAGES, 1024*NUM_IMAGES))

# Iterate over the image files and paste them onto the collage
for i, image_file in enumerate(image_files):
    if i >= NUM_IMAGES**2:
        break  # Stop after 9 images
    image = Image.open(image_file)
    x = (i % NUM_IMAGES) * 1024
    y = (i // NUM_IMAGES) * 1024
    collage.paste(image, (x, y))

# Display the collage until you find one you like
collage.show()