import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import torch
from phenobench import PhenoBench
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
from torch.cuda.amp import autocast
from matplotlib.colors import ListedColormap
from itertools import product

plt.rcParams["keymap.quit"] = ['ctrl+w', 'cmd+w', 'q']

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

def prediction_metrics(model_id, split, iter,  DEVICE, epoch_dir = ''):
    # load the model onto device
    model = torch.load(os.path.join('models', str(model_id),epoch_dir, 'model.pt'), map_location=torch.device(DEVICE)).to(DEVICE)
    model.eval()

    # load an image
    data = PhenoBench(os.path.join("data", "PhenoBench"), split = split , target_types=["semantics"])

    # Evaulation Metric
    multi_jaccard = MulticlassJaccardIndex(num_classes=3, average = None).to(DEVICE)

    ious = []
    for idx in tqdm(range(iter)):
        with autocast(): 
            # Format the image into the model input structure
            img = torch.tensor(np.transpose(np.array(data[idx]['image']), (2,0,1))).float().unsqueeze(0).to(DEVICE)
            ious.append(multi_jaccard(model.predict(img), torch.tensor(data[idx]['semantics']).unsqueeze(0).to(DEVICE)).cpu().tolist())
    
    return ious

def best_worst_image(results):
    # init dictionary of results
    best_worst = {'index':{}, 'iou':{}}
    keys = ['soil','crop','weed']

    # iterate over the List[List[]]
    for img_idx,result in enumerate(results):
        # iterate over the List[] of len 3
        for j, key in enumerate(keys):
            # check if we already have a best/worst value
            if 'best_' + key not in best_worst['iou']:
                best_worst['iou']['best_' + key] = 0
                best_worst['index']['best_' + key] = 'NA'

            # update index and iou if our current iou is better than the previous
            elif result[j] > best_worst['iou']['best_' + key]:
                best_worst['iou']['best_' + key] = result[j]
                best_worst['index']['best_' + key] = img_idx

            # check if we already have a best/worst value
            if 'worst_' + key not in best_worst['iou']:
                best_worst['iou']['worst_' + key] = 1
                best_worst['index']['worst_' + key] = 'NA'

            # update index and iou if our current iou is better than the previous
            elif result[j] < best_worst['iou']['worst_' + key]:
                best_worst['iou']['worst_' + key] = result[j]
                best_worst['index']['worst_' + key] = img_idx

    return best_worst

def plot_img_idx(model_id, split, img_idx, DEVICE, insert_title_text = '', epoch_dir= ''):
    model = torch.load(os.path.join('models', str(model_id),epoch_dir, 'model.pt'), map_location=torch.device(DEVICE)).to(DEVICE)
    model.eval()

    # load an image
    data = PhenoBench(os.path.join("data", "PhenoBench"), split = split , target_types=["semantics"])
    img = data[img_idx]['image']
    mask = np.array(data[img_idx]['semantics'])
    pred = model.predict(torch.tensor(np.transpose(np.array(img), (2,0,1))).float().unsqueeze(0).to(DEVICE)).cpu().numpy()[0]

    fig, axs = plt.subplots(1,4, figsize = (18,5))
    axs[0].imshow(img)
    axs[0].set_title('Image')

    axs[1].imshow(mask,  cmap = 'viridis')
    axs[1].set_title('Semantic Mask')

    axs[2].imshow(pred, cmap = 'viridis')
    axs[2].set_title('Prediction')

    axs[3].imshow((mask - pred).astype(int), cmap = 'viridis')
    axs[3].set_title('Semantic Mask Minus Prediction')

    fig.suptitle(f"Model ID: {model_id}, Split {split}, Image Index {img_idx}, {insert_title_text}")
    plt.show(block = False)

if __name__ == "__main__":
    #pixel_frequencies()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = '5265_balanced_weights_final_epochs'
    results = prediction_metrics(model_name, 'train', 1000,  DEVICE, 'epoch_0009')



    best_worst = best_worst_image(results)
    
    #perform, semant = 'best', 'crop'
    for perform, semant in product(['best', 'worst'], ['soil', 'crop', 'weed']):
        plot_img_idx(model_name, 'train', 
                     best_worst['index'][perform + '_' + semant], DEVICE, insert_title_text= f'{perform.capitalize()} {semant.capitalize()}: {best_worst['iou'][perform + '_' + semant]:.4f}',
                     epoch_dir = 'epoch_0009')

    
    str(input('Press enter to stop eda.py: '))