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
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import islice

def test():
    # C:\Users\menam\projects\phenobench\pheno_env\Scripts\activate
    # https://pytorch.org/get-started/locally/
    # is cuda avilable
    resize = 128
    batch_size = 16 # USE AS MUCH OF THE GPU AS POSSIBLE
    forward_all = True
    epochs = 4
    LR = .001
    plot_batch_loss = True # find a better way to plot the batch loss
    write_outcome = False
    w1,w2,w3 = .01, .2, .79

    transform = A.Compose([
                #A.LongestMaxSize(max_size=256),
                
                A.Resize(height = resize, width = resize)  # Resize the image to height 256 and width 256. OG: 1024, 1024
                #A.Normalize(),  # Normalize pixel values
                  # Convert the image to a PyTorch tensor
                ]) 
    
    # Set the random seed for CPU operations
    torch.manual_seed(42)

    # Set the random seed for CUDA operations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
    val_data = PhenoBench(r"data\PhenoBench", split = "val", target_types=["semantics"])

    # Create a DataLoader with batching and shuffling
    
    
    #print(f'Iterating through {len(train_data)} images with batch size {batch_size}: {len(train_data)//batch_size}')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last = True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last = True)

    #layer = DoubleConv(in_channels=3, out_channels=64).to(DEVICE) # ensure that weights and data are both on GPU
    model = UNET(in_channels = 3, out_channels=3).to(device=DEVICE)
    
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler() # automated mixed precision. Dynamically scale  between float16 and float32 stability and computation increases
    # loss_func https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
    # {0:soil, 1:crop, 2: weed}
    #loss_func = JaccardIndex(task="multiclass", num_classes=3).to(DEVICE) 
    # between 0 an 1. 1 is perfect score
    eval_func = MulticlassJaccardIndex(num_classes=3, average=None).to(DEVICE)
    
    loss_func = nn.CrossEntropyLoss(weight = torch.tensor([w1,w2,w3]).to(DEVICE)) # will more epochs help or will more skewed weights help
    # Define the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    fig1, ax1 = plt.subplots()
    if forward_all:
        epoch_evals = []
        epoch_loss = []

        for i in range(epochs):
            batch_loss = []
            for batch in tqdm( train_loader): # use islice(val_loader, 10) if writieing new code to test for bugs
                # context manager. AMP. 
                with autocast(): 
                    loss = loss_func(model(batch['images'].to(DEVICE)), batch['masks'].long().to(DEVICE))
                    loss = loss.requires_grad_()
                    batch_loss.append(loss)
                # Backward pass and optimization
                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update parameters
            print(f'Epoch {i} mean loss: {torch.mean(torch.stack(batch_loss))}')
            # convert batch loss into np array on the CPU
            batch_loss = torch.stack(batch_loss).detach().cpu().numpy()
            epoch_loss.append(batch_loss)

            if plot_batch_loss:
                ax1.plot(batch_loss, label = f'Epoch {i}')
                
            # evaluate on validation data
            multi_jaccard = MulticlassJaccardIndex(num_classes=3, average = None).to(DEVICE)
            batch_evals = []
            #print(f'Iterating through {len(val_data)} images with batch size {batch_size}: {len(val_data)//batch_size}')
            for batch in tqdm(val_loader): # use islice(val_loader, 10) if writieing new code to test for bugs
                
                with autocast(): # context manager. AMP. 
                    pred = model.predict(batch['images'].to(DEVICE))
                    batch_evals.append(multi_jaccard(pred, batch['masks'].float().to(DEVICE)))
            mean_epoch_evals = torch.mean(torch.stack(batch_evals, dim= 1),dim = 1)
            print(mean_epoch_evals)
            epoch_evals.append(mean_epoch_evals)

         # Add title and axis labels
        if plot_batch_loss:
            ax1.set_title('Batch Losses CE weight = (%s,%s,%s)'%(w1,w2,w3))
            ax1.set_xlabel('Batch Index')
            ax1.set_ylabel('Loss')
            ax1.legend()
            fig1.savefig("images/single_batch_losses_weight(%s,%s,%s).png"%(w1,w2,w3))
    
        #print(epoch_evals)
        #epoch_loss = epoch_loss.detach().cpu().numpy()
        #print(epoch_loss)
        fig2, ax2 = plt.subplots()
        epoch_loss_flat = [loss for epoch_list in epoch_loss for loss in epoch_list]
        ax2.plot(epoch_loss_flat)
        #ax2.plot(epoch_loss[1], label = 'Epoch 1')
        # Add title and axis labels
        ax2.set_title('Batch Losses CE weight = (%s,%s,%s)'%(w1,w2,w3))
        ax2.set_xlabel('Batch Index')
        ax2.set_ylabel('Loss')
        #ax2.legend()
        fig2.savefig("images/epochs_batch_losses_weight(%s,%s,%s).png"%(w1,w2,w3))

        #print(eval_loss)
        # I'd like to plot the evaluation loss for each epochand the training loss for each epoch
        epoch_evals = np.array([[tensor.detach().cpu().numpy() for tensor in eval_list] for eval_list in epoch_evals])
        
        fig3, ax3 = plt.subplots()
        # iterate over the second axis
        for i in range(epoch_evals.shape[1]):

            ax3.plot(epoch_evals[:,i], label = 'JaccardIndex')
        ax3.plot(np.mean(np.array(epoch_loss), axis = 1), label = 'Loss')
        #plt.show(fig3)
        ax3.set_title('validation and loss by epoch weight(%s,%s,%s)'%(w1,w2,w3))
        ax3.set_xlabel('Epoch Index')
        ax3.set_ylabel('Loss and JaccardIndex')
        ax3.legend()
        fig3.savefig("images/eval_loss_by_epoch_weight(%s,%s,%s).png"%(w1,w2,w3))
        
            #hours, minutes, seconds = int(time_diff // 3600), int(time_diff // 60), int(time_diff % 60)
            
            # if write_outcome:
            #     with open('output.txt', 'a') as file:
            #         file.write(f'\n#{resize: < 10}| {batch_size: <10}| {forward_all: <10}| yes    | {hours}:{minutes}:{seconds:.02f}  | {epochs: <10} #')
                
            
    else: 

         # Get the first batch from the iterator
        first_batch = next(iter(train_loader))
        print(first_batch['images'].shape, first_batch['masks'].shape)
        #first_batch = next(train_loader_iter)
        batch_images = first_batch['images'].to(DEVICE)
        batch_masks = first_batch['masks'].to(DEVICE)
        
        soft = model(batch_images)
        preds = torch.argmax(soft, dim=1)
        evals = []
        multi_jaccard = MulticlassJaccardIndex(num_classes=3, average = None).to(DEVICE)
        for p,m in zip(preds, batch_masks):
            evals.append(multi_jaccard(p, m))
        
        evals = torch.mean(torch.stack(evals, dim= 1),dim = 1)
        print(evals)
        print(multi_jaccard(preds, batch_masks)) # Additional dimension ... will be flattened into the batch dimension.
if __name__ == "__main__":
    test()