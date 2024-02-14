from phenobench import PhenoBench
#from dataset import get_batch_idx, custom_collate
from models import DoubleConv, UNET
import numpy as np
import torch
import torch.nn as nn
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm




def test():
    # C:\Users\menam\projects\phenobench\pheno_env\Scripts\activate
    # https://pytorch.org/get-started/locally/
    # is cuda avilable
    resize = 64
    batch_size = 32
    forward_all = True
    epochs = 1

    transform = A.Compose([
                #A.LongestMaxSize(max_size=256),
                
                A.Resize(height = resize, width = resize)  # Resize the image to height 256 and width 256. OG: 1024, 1024
                #A.Normalize(),  # Normalize pixel values
                  # Convert the image to a PyTorch tensor
                ]) 
    
    
    def custom_collate(batch):
        
        
        # Extract images and labels from the batch
        #images = np.array([item['image'] for item in batch])
        #masks = np.array([item['semantics'] for item in batch])
        # images , masks = [] , []
        # for item in batch:
            
        #     transformed = transform(images = np.array(item['image']), masks = np.array(item['semantics']))
        #     images.append(transformed['images'])
        #     batch.append(transformed['masks'])
        transformed = [transform(image=np.array(item['image']), mask = np.array(item['semantics'])) for item in batch]
        images = [item['image'] for item in transformed]
        masks = [item['mask'] for item in transformed]

        
        return {'images': torch.tensor(np.transpose(np.array(images), (0,3,1,2))).float(), 'masks': torch.tensor(np.array(masks))}
        #return {'images': images, 'masks': masks}
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
   

    # phenobenchs dataloader. It sits on the directroy and only loads when .__getitem__ is called
    train_data = PhenoBench(r"data\PhenoBench", split = "train", target_types=["semantics"])
    
    #transformed = transform(images = train_data, masks = np.array(train_data[0]['semantics']))
    #print(transformed['images'])
    
     
    #         Table of Hyperparamters           #
    # ------------------------------------------------#
    # Resize | Batch_size| all_forward      | sucess | time   | epochs  #
    # 512    | 8         | single           | yes    |        #
    # 1024   | 8         | single           | no     |        #
    # 1024   | 4         | single           | yes    |        #
    # 1024   | 2         | all              | no     |        # 
    # 512    | 2         | all              | yes    | 3:03   #
    # 128    | 2         | True             | yes    | 0.0:59.173381   #
    #  128   | 16        | True    | yes    | 0:54.42   #
    # 128 |32        |1         | yes    | 0:53.05  | 1          #
    # 256      | 32        | 1         | yes    | 6:55.82  | 1          #

    # Create a DataLoader with batching and shuffling
    
    
    print(f'Itertions through {len(train_data)} images with batch size {batch_size}: {len(train_data)//2}')
    print('# Resize | Batch_size')
    print(f' %-6d | %-9d'%(resize, batch_size))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last = True)

    layer = DoubleConv(in_channels=3, out_channels=64).to(DEVICE) # ensure that weights and data are both on GPU

    
    
    model = UNET(in_channels = 3, out_channels=1).to(device=DEVICE)

    # Get the first batch from the iterator
    # first_batch= next(iter(train_loader))
    # print(first_batch['images'].shape, first_batch['masks'].shape)
    # #first_batch = next(train_loader_iter)
    # batch_images = first_batch['images'].to(DEVICE)
    # batch_masks = first_batch['masks'].to(DEVICE)
    #print(batch_images.shape)
    #transformed = transform(image=batch_images['images'], mask = batch_images['masks'])
    #batch_masks
    shapes = []
    
    
    old_time = time.time()
    for batch in tqdm(train_loader):
        with autocast():
            
            #print(layer(batch_images).shape)
    #     # we need to rezie before passing to unet or figure out how unet fcan be more effiecint
            #shapes.append(model(batch['images'].to(DEVICE)).shape)
            preds = model(batch['images'].to(DEVICE))
            # loss_func
           # batch_time =  time.time() - current_time
           # batch_times.append(batch_time)
            #print(batch_time)
    # Calculate the time difference
    time_diff = time.time() - old_time

    # Convert time difference to minutes and seconds
    minutes = int(time_diff // 60)
    seconds = time_diff % 60
    print("Batch Complete")
    print(f'#{resize: < 10}| {batch_size: <10}| {forward_all: <10}| yes    | {minutes}:{seconds:.02f}  | {epochs: <10} #')
    with open('output.txt', 'a') as file:
        file.write(f'\n#{resize: < 10}| {batch_size: <10}| {forward_all: <10}| yes    | {minutes}:{seconds:.02f}  | {epochs: <10} #')
    

    # for batch in train_loader:
    #     batch_images = torch.tensor(np.transpose(batch['images'], (0,3,1,2))).float().to(DEVICE)
    #     batch_masks = batch['masks'].to(DEVICE)
    
    #
    
    # for batch in batch_images:
    #      pred = model(batch)
    #      print(pred.shape)


if __name__ == "__main__":
    test()