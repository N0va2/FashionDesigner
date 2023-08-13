from PIL import Image
import os

# Directory containing your images
image_dir = 'Designer/ClothesFits/iCloudPhotos'

# List to store dimensions
widths = []
heights = []

# Iterate over each image
for image_name in os.listdir(image_dir):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # add any other image file types if necessary
        with Image.open(os.path.join(image_dir, image_name)) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)

# Calculate mean width and height
mean_width = sum(widths) / len(widths)
mean_height = sum(heights) / len(heights)

print(f"Average Width: {mean_width}")
print(f"Average Height: {mean_height}")
