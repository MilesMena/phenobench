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
import json
# I despise np arrays that display in scientific notaton
np.set_printoptions(formatter={'float_kind':'{:f}'.format})



# TODO: save checkoutpoint function
    
# TODO: Plotting Function(s)

# TODO: 

def main():
    # Activate my python environment for this task:                       C:\Users\menam\projects\phenobench\pheno_env\Scripts\activate
    # Install the most stable version of pytorch with GPU configuration:  https://pytorch.org/get-started/locally/

    ############## Hyperparameters #################
    RESIZE = 128
    BATCH_SIZE = 4 # I hear you are supposed to use as much GPU as possible, but batch size affects the loss propagations. Look into this tradeoff 
    EPOCHS = 1
    LR = .001
    # use GPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # For Ryan's puny mac:
    DEVICE = 'mps' if torch.backends.mps.is_available() else DEVICE
    print(DEVICE)

    # Set the random seed for CPU operations
    torch.manual_seed(42)
    # Set the random seed for CUDA operations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    LOSS_WEIGHT = torch.ones(3).to(DEVICE) # 3 is the number of classes for this task
    # calculate weights by processing dataset histogram balancing by class 
    LOSS_WEIGHT[0], LOSS_WEIGHT[1], LOSS_WEIGHT[2] = .1,2,100                                # CLASS LABELS: {0:soil, 1:crop, 2: weed}

    ############# Init the Phenobench DataLoader #################
    # Phenobench's DataLoader sits on top the directroy and only loads when .__getitem__ is called
    # ex: train_data[image_index]['image']
    DATA_PATH = os.path.join("data", "PhenoBench") # OS independent path
    train_data = PhenoBench(DATA_PATH, split = "train", target_types=["semantics"])
    val_data = PhenoBench(DATA_PATH, split = "val", target_types=["semantics"])

    ################ Data Augmentation ################
    # We resize so that our GPU's can handle the number of parameters 
    # TODO: look into applying more augmentations other than resize
    transform = A.Compose([
                A.Resize(height = RESIZE, width = RESIZE)  # original images size:  (1024, 1024)
                ]) 
    
    
    ################# PyTorch DataLoader ##################
    # Does this load ALL of the training images into memory, or only those collected for the batch that is currently training
    def custom_collate(batch):
        # Collate customizes how individual samples are grouped together https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3
        # The phenobench library provides an efficient way to load data, but it doesn't batch the images
        # So we write a custom collation function to neatly pass the data to DataLoader, and we transform the data with albumentations here because
        # it helps save memory loading in a rezied image.
        # This function could possibly go outside main(), but that changes the relationship to the function transform. Look into this
        transformed = [transform(image=np.array(item['image']), mask = np.array(item['semantics'])) for item in batch]
        images = [item['image'] for item in transformed]
        masks = [item['mask'] for item in transformed]

        
        return {'images': torch.tensor(np.transpose(np.array(images), (0,3,1,2))).float(), 'masks': torch.tensor(np.array(masks))}

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, drop_last = True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, drop_last = True)


    ################# Init the Model #####################
    #layer = DoubleConv(in_channels=3, out_channels=64).to(DEVICE) # ensure that weights and data are both on GPU
    model = UNET(in_channels = 3, out_channels=3).to(device=DEVICE)
    
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler() # automated mixed precision. Dynamically scale between float16 and float32 stability and computation increases during back prop
    
    eval_func = MulticlassJaccardIndex(num_classes=3, average=None).to(DEVICE) # https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
    
    loss_func = nn.CrossEntropyLoss(weight = LOSS_WEIGHT) # will more epochs help or will more skewed weights help
    
    # Define the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr = LR)


    ############### Train the Model ###############
    eval_loss = []
    epoch_evals = []
    epoch_loss = []

    for i in range(EPOCHS):
        batch_loss = []
        for batch in tqdm(train_loader): # # testing code: tqdm(islice(train_loader,8))  Training model:  tqdm(train_loader)
        
            ############ Compute Loss for each Batch ###########
            with autocast(): # context manager. AMP. Uses float16 and float32 when appropriate to decrease computations expense 
                loss = loss_func(model(batch['images'].to(DEVICE)), batch['masks'].long().to(DEVICE))
                loss = loss.requires_grad_()
                batch_loss.append(loss)
            ############ Backpropagate the Loss ##############
            optimizer.zero_grad()  # Zero the gradients, so we only back prop this batch not this batch and all of the batches before it
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters

        print(f'Epoch {i} mean loss: {torch.mean(torch.stack(batch_loss))}')


        # convert batch loss into np array on the CPU
        batch_loss = torch.stack(batch_loss).detach().cpu().numpy()
        epoch_loss.append(batch_loss)

     
            
        ############### Evaluate on Validation ###########
        # I am unsure if we should be evaluating or applying the loss function
        # Evaluation easily shows the performance on all of the class, but I think generally people look at the validation loss
        # be careful to avoid overfitting on the validation set
        multi_jaccard = MulticlassJaccardIndex(num_classes=3, average = None).to(DEVICE)
        val_loss = []
        val_iou = []
        for batch in tqdm(val_loader): # testing code: tqdm(islice(train_loader),8)  Training model:  tqdm(val_loader)
            
            with autocast(): # context manager. AMP. 
                # For cross entropy loss we pass the raw logits compared to the labels
                logits = model(batch['images'].to(DEVICE)) 
                # loss.item returns a python float, and without it we put a bunch of tensors on the GPU which takes up memory 
                val_loss.append(loss_func(logits, batch['masks'].long().to(DEVICE)).item())
                # iou.detech().cpu().tolist() first removes grad_requried = True, then moves to the cpu then converts to a list
                val_iou.append(multi_jaccard(model.predict_from_logits(logits), batch['masks'].long().to(DEVICE)).detach().cpu().tolist())

        mean_val_loss = np.array(val_loss).mean()
        mean_val_iou = np.array(val_iou).mean(axis = 0) # mean of the columns

        
        print(f"Val Mean Loss: {mean_val_loss}, Validation IoU: {mean_val_iou}")
        #np.savetxt('validation_prediction.txt', pred.reshape(-1, 1).detach().cpu().numpy())


    #    if plot_batch_loss:
    #         ax1.plot(batch_loss, label = f'Epoch {i}')
    #     # Add title and axis labels
    # fig1, ax1 = plt.subplots()
    # if plot_batch_loss:
    #     ax1.set_title('Batch Losses CE weight = (%s,%s,%s)'%(w1,w2,w3))
    #     ax1.set_xlabel('Batch Index')
    #     ax1.set_ylabel('Loss')
    #     ax1.legend()
    #     fig1.savefig("images/single_batch_losses_weight(%s,%s,%s).png"%(w1,w2,w3))

    # #print(epoch_evals)
    # #epoch_loss = epoch_loss.detach().cpu().numpy()
    # #print(epoch_loss)
    # fig2, ax2 = plt.subplots()
    # epoch_loss_flat = [loss for epoch_list in epoch_loss for loss in epoch_list]
    # ax2.plot(epoch_loss_flat)
    # #ax2.plot(epoch_loss[1], label = 'Epoch 1')
    # # Add title and axis labels
    # ax2.set_title('Batch Losses CE weight = (%s,%s,%s)'%(w1,w2,w3))
    # ax2.set_xlabel('Batch Index')
    # ax2.set_ylabel('Loss')
    # #ax2.legend()
    # fig2.savefig("images/epochs_batch_losses_weight(%s,%s,%s).png"%(w1,w2,w3))

    # #print(eval_loss)
    # # I'd like to plot the evaluation loss for each epochand the training loss for each epoch
    # epoch_evals = np.array([[tensor.detach().cpu().numpy() for tensor in eval_list] for eval_list in epoch_evals])
    
    # fig3, ax3 = plt.subplots()
    # # iterate over the second axis
    # for i in range(epoch_evals.shape[1]):

    #     ax3.plot(epoch_evals[:,i], label = 'JaccardIndex')
    # ax3.plot(np.mean(np.array(epoch_loss), axis = 1), label = 'Loss')
    # #plt.show(fig3)
    # ax3.set_title('validation and loss by epoch weight(%s,%s,%s)'%(w1,w2,w3))
    # ax3.set_xlabel('Epoch Index')
    # ax3.set_ylabel('Loss and JaccardIndex')
    # ax3.legend()
    # fig3.savefig("images/eval_loss_by_epoch_weight(%s,%s,%s).png"%(w1,w2,w3))
        
    # TODO: We need to save the model, but I also think we should save the results and the hyperparameters by appending to a csv file: training_loss, val_loss, training_time, training_time per batch, IoU, etc
    model_parameters = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "loss_weight": LOSS_WEIGHT.tolist(),
        "resize": RESIZE,
        "device": DEVICE,
        "mean_val_loss": mean_val_loss,
        "mean_val_iou": mean_val_iou.tolist()
    }
    save_model(model, model_parameters)


def save_model(model, model_stats={}):
    model_id = str(np.random.randint(10000))
    model_path = os.path.join("models", model_id)
    os.makedirs(model_path)
    torch.save(model, os.path.join(model_path, "model.pt"))
    with open(os.path.join(model_path, "model_stats.json"), "w") as json_file:
        json.dump(model_stats, json_file, indent=4)
    print(f"Model saved at {model_path}")

                
if __name__ == "__main__":
    main()