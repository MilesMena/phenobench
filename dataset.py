# import os # for filepath and retrival
# from PIL import Image # for handling .png files https://pillow.readthedocs.io/en/stable/reference/Image.html
# import numpy as np # handling array 
# from torch.utils.data import Dataset # for handling iterable datasets in torch

# class PlantDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform = None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = os.listdir(image_dir) # returns a list of the names of entries in dir provided

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         # RBG image and it's mask have the same name, different dir
#         img_path = os.path.join(self.image_dir, self.images[index])
#         mask_path = os.path.join(self.mask_dir, self.image[index])
#         image = np.array(Image.open(img_path).convert("RGB")) # convert returns a convert copy of the image for the given mode
#         mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32) # {modes:  {RGB: 3x8 bit (true color), L: 8 bit (grayscale 0-255)https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes

from phenobench import PhenoBench # pip install phenobench
# Phenobench returns getitem with sample['image'] as a PIL.Image object not an array, 
# so I copy and pasted the code and wrapped sample['image'] in np.array
# we could keep it as a PIL.Image object to keep some of the functionalities of PIL.Image,
# but for batching it is easier.
#from SDK import PhenoBench 
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
import torch

def get_batch_idx(train_data, batch_size = 16):
    
    idx = np.arange(len(train_data))
    bathes_idx = []
    for i in range(len(idx) // batch_size):
        random_idx = np.random.choice(idx, size = batch_size, replace = False)
        idx = np.delete(idx, np.where(np.isin(idx, random_idx)))
        bathes_idx.append(random_idx)
    batches_idx = np.array(bathes_idx)
    return batches_idx

def custom_collate(batch):
        # Extract images and labels from the batch
        images = np.array([item['image'] for item in batch])
        masks = torch.tensor(np.array([item['semantics'] for item in batch]))
        
        # Return batched images and labels
        return {'images': images, 'masks': masks}
    





    
