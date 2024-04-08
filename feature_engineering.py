import numpy as np
from phenobench import PhenoBench
import os
import pickle
import matplotlib.pyplot as plt


DATA_PATH = os.path.join("data", "PhenoBench") # OS independent path
YOLO_PATH = os.path.join("data", "yolo_labeled_images") # OS independent path

SAVE_PATH = os.path.join("data", "sandwich_images") # OS independent path

SHOW_IMAGES = False

# EXTRA_CHANNELS = { # These feature channels come from this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460962
#     "ExG" : lambda pix: np.dot(pix[:3], [-1, 2, -1]), #2*pix[1] - pix[0] - pix[2],
#     "ExR": lambda pix: np.dot(pix[:3], [1.4, -1, 0]),
#     "CIVE": lambda pix: np.dot(pix[:3], [-0.441, 0.881, -0.385]) - 18.78745,
#     "NDI": lambda pix: (pix[1] - pix[0])/(pix[1] + pix[0])
# }




def show_channels(image):

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs[0, 0].imshow(image[:,:,0], cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title('R Channel')
    axs[0, 1].imshow(image[:,:,1], cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title('G Channel')
    axs[0, 2].imshow(image[:,:,2], cmap='gray', vmin=0, vmax=255)
    axs[0, 2].set_title('B Channel')
    axs[0, 3].imshow(image[:,:,3], cmap='gray')
    axs[0, 3].set_title('ExG Channel')
    axs[1, 0].imshow(image[:,:,4], cmap='gray')
    axs[1, 0].set_title('ExR Channel')
    axs[1, 1].imshow(image[:,:,5], cmap='gray')
    axs[1, 1].set_title('CIVE Channel')
    axs[1, 2].imshow(image[:,:,6], cmap='gray')
    axs[1, 2].set_title('NDI Channel')
    axs[1, 3].imshow(image[:,:,7], cmap='gray')
    axs[1, 3].set_title('YOLO Channel')
    plt.tight_layout()
    plt.show()



def create_features():
    for split in ('train', 'val', 'test'):
        data = PhenoBench(DATA_PATH, split = split, target_types=["semantics"])

        new_data = []
        for i in range(len(data)):
            print(f"Processing image {split}/{data[i]['image_name']}")
            image = np.array(data[i]['image'])

            # open an image and convert it to a numpy array
            yolo_image = plt.imread(os.path.join(YOLO_PATH, split, data[i]['image_name']))
            yolo_image = np.array(yolo_image)
            yolo_image = yolo_image[:,:,0]



            R = image[:,:,0]
            G = image[:,:,1]
            B = image[:,:,2]
            ExG = 2*G - R - B
            ExR = 1.4*R - G
            CIVE = -0.441*R + 0.881*G - 0.385*B - 18.78745
            NDI = (G - R)/(G + R + 1) # +1 to avoid division by zero
            YOLO = yolo_image

            new_image = np.stack([R, G, B, ExG, ExR, CIVE, NDI, YOLO], axis = 2)
            new_image = np.clip(new_image, 0, 255) # clipping to avoid negative and infinite values. Maybe scaling would be easier, but that should be done channelwise
            new_image = new_image.astype(np.uint8)

            if SHOW_IMAGES:
                show_channels(new_image)

            # display each of the image channels in a subplot

            npy_name = data[i]['image_name'].split(".")[0] + ".npy"

            with open(os.path.join(SAVE_PATH, split, npy_name), 'wb') as f:
                np.save(f, new_image)



def display_images():
    for split in ('train', 'val', 'test'):
        data = PhenoBench(DATA_PATH, split = split, target_types=["semantics"])

        for i in range(len(data)):
            npy_name = data[i]['image_name'].split(".")[0] + ".npy"
            print(f"Displaying image {split}/{npy_name}")
            image = np.load(os.path.join(SAVE_PATH, split, npy_name))
            show_channels(image)


if __name__ == "__main__":

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        os.makedirs(os.path.join(SAVE_PATH, "train"))
        os.makedirs(os.path.join(SAVE_PATH, "val"))
        os.makedirs(os.path.join(SAVE_PATH, "test"))
        create_features()
    else:
        print("Features already created, skipping creation.")
        display_images()