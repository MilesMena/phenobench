from phenobench import PhenoBench
from dataset import get_batch_idx
from models import DoubleConv, UNET
import numpy as np
import torch
import torch.nn as nn

def test():
    # phenobenchs dataloader. It sits on the directroy and only loads when .__getitem__ is called
    train_data = PhenoBench("data\PhenoBench",split = "train", target_types=["semantics"])
    # returns a list of indices in batches [[1,100, 23, 24],[2,11,25, 12]....]
    batch_idx = get_batch_idx(train_data, batch_size=2)
    # from our indices, get the image np array and mask from train_data. 5d array (batch_num, samples, channels, height, width)
    batch_images, batch_masks = [], []
    for batches in batch_idx[:2]:
        images = []
        masks = []
        for idx in batches:
            images.append(np.array(train_data[idx]['image']) )
            masks.append(train_data[idx]['semantics'])   
        batch_images.append(images)
        batch_masks.append(masks) 

    # print(np.array(batch_images).shape)
    # 
    batch_images = torch.tensor(np.transpose(np.array(batch_images), (0,1,4,2,3))).float()
    batch_masks = torch.tensor(np.array(batch_masks))

    #layer = DoubleConv(in_channels=3, out_channels=64)
    model = UNET(in_channels = 3, out_channels=1)
    
    for batch in batch_images:
        pred = model(batch)
        print(pred.shape)

    
   
    

if __name__ == "__main__":
    test()