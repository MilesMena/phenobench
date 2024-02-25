import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def pixel_frequencies():


    # Specify the directory containing your images
    DISPLAY_IMAGES = False
    # Initialize counters
    count_dirt = []
    count_plant = []
    count_weed = []
    total_files = 0
    # Iterate over each image in the directory
    for path in [os.path.join("data", "PhenoBench", "train", "semantics"), os.path.join("data", "PhenoBench", "val", "semantics")]:
        for filename in os.listdir(path):
            if filename.endswith('.png') or filename.endswith(".jpg"):  # Add or modify to suit your image file types
                # Open the image file
                with Image.open(os.path.join(path, filename)) as img:
                    # Convert the image data to a numpy array
                    data = np.array(img)
                    # Increment the counters
                    count_dirt.append(Decimal(int(np.sum(data == 0))))
                    count_plant.append(Decimal(int(np.sum(data == 1) + np.sum(data == 3))))
                    count_weed.append(Decimal(int(np.sum(data == 2) + np.sum(data == 4))))
                    total_files += 1
            # display the image with matplotlib
                    if DISPLAY_IMAGES:
                        plt.subplot(1, 2, 1)
                        plt.imshow(data)
                        plt.title('Mask')
                        plt.subplot(1, 2, 2)
                        plt.imshow(Image.open(os.path.join(path, "..", "images", filename)))
                        plt.title('Image')
                        plt.show()

    total_pixels = sum(count_dirt) + sum(count_plant) + sum(count_weed)

    print(f"average dirt percentage: {100*sum(count_dirt)/total_pixels}")
    print(f"average plant percentage: {100*sum(count_plant)/total_pixels}")
    print(f"average weed percentage: {100*sum(count_weed)/total_pixels}")

    dirt_percentage, plant_percentage, weed_percentage= [], [], []

    for dirt, plant, weed in zip(count_dirt, count_plant, count_weed):
        dirt_percentage.append(100 * dirt / (dirt + plant + weed))
        plant_percentage.append(100 * plant / (dirt + plant + weed))
        weed_percentage.append(100 * weed / (dirt + plant + weed))

    # Plotting histograms
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.hist(dirt_percentage, bins=20, color='brown')
    plt.xlabel('Dirt')
    plt.ylabel('Count')
    plt.title('Dirt Classification')

    plt.subplot(1, 3, 2)
    plt.hist(plant_percentage, bins=20, color='green')
    plt.xlabel('Plant')
    plt.ylabel('Count')
    plt.title('Plant Classification')

    plt.subplot(1, 3, 3)
    plt.hist(weed_percentage, bins=20, color='purple')
    plt.xlabel('Weed')
    plt.ylabel('Count')
    plt.title('Weed Classification')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pixel_frequencies()