from phenobench import PhenoBench
import os, csv, sys, json
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
from models import UNET
import cv2
# DISUSED IMPORTS
# import time
# from dataset import get_batch_idx, custom_collate
# from torchmetrics import JaccardIndex
# from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
# from itertools import islice

# I despise np arrays that display in scientific notaton
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

    
# TODO: Plotting Function(s)
############## Hyperparameters #################
RESIZE = 1024
BATCH_SIZE = 1 # I hear you are supposed to use as much GPU as possible, but batch size affects the loss propagations. Look into this tradeoff 
EPOCHS = 100
LR = .0001
EVALUATE_IN_LOOP = True
TRAIN_PERCENTAGE = 1 # Should nominally be 1, but for testing purposes we can set it to a fraction of the dataset
VAL_PERCENTAGE = .25 # Should nominally be 1, but for testing purposes we can set it to a fraction of the dataset
SAVE_EVERY = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = os.path.join("data", "PhenoBench") # OS independent path
# calculate weights by processing dataset histogram balancing by class 
LOSS_WEIGHT = ((1/88.45), (1/11.03), (1/.5))                       # CLASS LABELS: {0:soil, 1:crop, 2: weed}
EXTRA_CHANNELS = { # These feature channels come from this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460962
    "ExG" : lambda pix: np.dot(pix[:3], [-1, 2, -1]), #2*pix[1] - pix[0] - pix[2],
    "ExR": lambda pix: np.dot(pix[:3], [1.4, -1, 0]),
    "CIVE": lambda pix: np.dot(pix[:3], [-0.441, 0.881, -0.385]) - 18.78745,
    "NDI": lambda pix: (pix[1] - pix[0])/(pix[1] + pix[0])
}
ADD_EDGE_CHANNEL = True

def main(model_name):
    # Activate my python environment for this task:                       C:\Users\menam\projects\phenobench\pheno_env\Scripts\activate
    # Install the most stable version of pytorch with GPU configuration:  https://pytorch.org/get-started/locally/
    # use GPU
    set_device()
    train_loader, val_loader = data_loaders()

    ################# Init the Model #####################
    #layer = DoubleConv(in_channels=3, out_channels=64).to(DEVICE) # ensure that weights and data are both on GPU
    model = UNET(in_channels = (len(EXTRA_CHANNELS)+3), out_channels=3).to(device=DEVICE)
    
    # model = deeplabv3_resnet50(weights=None).to(DEVICE)
    # model.classifier[4] = nn.Conv2d(256, out_channels= 3, kernel_size=(1,1), stride = (1,1)) # previous model: Linear(in_features=2048, out_features=1000, bias=True)
    model.to(DEVICE)
    

    ############## Load a Saved Model ####################
    # model = models.resnet18()
    # # Load the saved model state dictionary
    # saved_model_path = 'saved_model.pth'
    # saved_model_state_dict = torch.load(saved_model_path)
    # # Load the saved model state dictionary into your model
    # model.load_state_dict(saved_model_state_dict)
    # model.to(DEVICE)

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler() # automated mixed precision. Dynamically scale between float16 and float32 stability and computation increases during back prop
    
    # Loss function. Why do we choose ot use CrossEntropy?
    loss_func = nn.CrossEntropyLoss(weight = LOSS_WEIGHT) # will more epochs help or will more skewed weights help
    
    # Define the Adam optimizer. Why do we choose to use the Adam optimizer?
    optimizer = optim.Adam(model.parameters(), lr = LR)

    ############### Train the Model ###############
    for i in range(EPOCHS):
        batch_loss = []
        for batch in tqdm(train_loader, colour="green"): # # testing code: tqdm(islice(train_loader,8))  Training model:  tqdm(train_loader)
            model.train()
            optimizer.zero_grad()  # Zero the gradients, so we only back prop this batch not this batch and all of the batches before it

            ############ Compute Loss for each Batch ###########
            with autocast(): # context manager. AMP. Uses float16 and float32 when appropriate to decrease computations expense 
                #loss = loss_func(model(batch['images'].to(DEVICE)), batch['masks'].long().to(DEVICE))
                loss = loss_func(model(batch['images'].to(DEVICE)), batch['masks'].long().to(DEVICE)) # pytroch.models.segmentation returns an OrderedDict with 'out' as the only key
                loss = loss.requires_grad_()
            batch_loss.append(loss.item())
            ############ Backpropagate the Loss ##############
            # if loss.item() == float('nan'):
            #     mean_val_loss, mean_val_iou = evaluate_validation(model, val_loader, DEVICE, loss_func)
            #     break
            
            scaler.scale(loss).backward()  # Compute gradients
            scaler.step(optimizer) # Update parameters
            scaler.update()

        

        print(f'Epoch {i} mean loss: {sum(batch_loss)/len(batch_loss)}')

        if ((i + 1) % SAVE_EVERY == 0) or (i == EPOCHS - 1): 
            #mean_val_loss, mean_val_iou = evaluate_validation(model, val_loader, DEVICE, loss_func)
            # ############### Evaluate on Validation ###########
            # # I am unsure if we should be evaluating or applying the loss function
            # # Evaluation easily shows the performance on all of the class, but I think generally people look at the validation loss
            # # be careful to avoid overfitting on the validation set
            if EVALUATE_IN_LOOP:
                mean_val_loss, mean_val_iou = evaluate_validation(model, val_loader, DEVICE, loss_func)
                print(f"Val Mean Loss: {mean_val_loss}, Validation IoU: {mean_val_iou}")
            else:
                mean_val_loss, mean_val_iou = 'NA', 'NA'

            
            model_parameters = {
                        "model_name": model_name,
                        "batch_size": BATCH_SIZE,
                        "epochs": i,
                        "learning_rate": LR,
                        "loss_weight": LOSS_WEIGHT.tolist(),
                        "resize": RESIZE,
                        "device": DEVICE,
                        "mean_val_loss": mean_val_loss,
                        "mean_val_iou": mean_val_iou,
                    }

            save_model(model, model_parameters)

        ################## MODEL STATISTICS #############
        # # convert batch loss into np array on the CPU
        # batch_loss = torch.stack(batch_loss).detach().cpu().numpy()
        # epoch_loss.append(batch_loss)
        # if flag:
        #     break



def set_device():
    global DEVICE, LOSS_WEIGHT
    torch.manual_seed(42) # Set the random seed for CPU operations
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        torch.cuda.manual_seed_all(42) # Set the random seed for CUDA operations (if available)
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
    print("DEVICE:", DEVICE)
    LOSS_WEIGHT = torch.tensor(LOSS_WEIGHT).to(DEVICE) # 3 is the number of classes for this task


def add_channels(image):
        if ADD_EDGE_CHANNEL:
            image = add_edge_channel(image)
        for _, fun in EXTRA_CHANNELS.items():
            applied = np.apply_along_axis(fun, 2, image)
            total = np.expand_dims(applied, axis=2)
            image = np.concatenate((image, total),axis=2)
        return image

def add_edge_channel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=350, threshold2=400)
    edges = edges / edges.max()
    edges = np.expand_dims(edges, axis=2)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(gray, cmap='gray')
    # ax[0].set_title('Image')
    # ax[1].imshow(edges, cmap='gray')
    # ax[1].set_title('Edges')
    # plt.show()
    return np.concatenate((image, edges), axis=2)
    


def data_loaders():
    ############# Init the Phenobench DataLoader #################
    # Phenobench's DataLoader sits on top the directroy and only loads when .__getitem__ is called
    # ex: train_data[image_index]['image']
    train_data = PhenoBench(DATA_PATH, split = "train", target_types=["semantics"])
    val_data = PhenoBench(DATA_PATH, split = "val", target_types=["semantics"])

    print(np.array(train_data[0]['image']))
    if len(EXTRA_CHANNELS) > 0:
        for image in tqdm(train_data, colour="blue"):
            image['image'] = add_channels(np.array(image['image']))
        for image in tqdm(val_data, colour="blue"):
            image['image'] = add_channels(np.array(image['image']))
    print(train_data[0]['image'])


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
        transformed = [transform(image=item['image'], mask = np.array(item['semantics'])) for item in batch]
        images = [item['image'] for item in transformed]
        masks = [item['mask'] for item in transformed]
        return {'images': torch.tensor(np.transpose(np.array(images), (0,3,1,2))).float(), 'masks': torch.tensor(np.array(masks))}

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=custom_collate, drop_last = True, sampler=RandomSampler(train_data, replacement=False, num_samples=int(len(train_data)*TRAIN_PERCENTAGE)))
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=custom_collate, drop_last=True, sampler=RandomSampler(val_data, replacement=False, num_samples=int(len(val_data)*VAL_PERCENTAGE)))

    return train_loader, val_loader

def save_model(model, model_stats={}):
    model_path = os.path.join("models", model_stats["model_name"])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        header = ['Epoch', 'Mean Loss', 'Mean IOU']
        with open(os.path.join(model_path, 'history.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    current_data = [model_stats["epochs"], model_stats["mean_val_loss"], model_stats["mean_val_iou"]]
    with open(os.path.join(model_path, 'history.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(current_data)


    epoch_path = os.path.join(model_path, f"epoch_{str(model_stats['epochs']).zfill(4)}")
    os.makedirs(epoch_path)
    torch.save(model, os.path.join(epoch_path, "model.pt"))
    with open(os.path.join(epoch_path, "model_stats.json"), "w") as json_file:
        json.dump(model_stats, json_file, indent=4)
    print(f"Model saved at {epoch_path}")

def evaluate_validation(model, val_loader, DEVICE, loss_func):
        model.eval()
        multi_jaccard = MulticlassJaccardIndex(num_classes=3, average = None).to(DEVICE) # https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
        val_loss = []
        val_iou = []
        print('VALIDATION: ')
        for batch in tqdm(val_loader, colour="red"): # testing code: tqdm(islice(train_loader),8)  Training model:  tqdm(val_loader)
            
            with autocast(): # context manager. AMP. 
                # For cross entropy loss we pass the raw logits compared to the labels
                logits = model(batch['images'].to(DEVICE)) 

                # loss.item returns a python float, and without it we put a bunch of tensors on the GPU which takes up memory 
                val_loss.append(loss_func(logits, batch['masks'].long().to(DEVICE)).item())
                # iou.detech().cpu().tolist() first removes grad_requried = True, then moves to the cpu then converts to a list
                val_iou.append(multi_jaccard(model.predict_from_logits(logits), batch['masks'].long().to(DEVICE)).detach().cpu().tolist())

        mean_val_loss = np.array(val_loss).mean()
        mean_val_iou = np.array(val_iou).mean(axis = 0).tolist() # mean of the columns

        return mean_val_loss, mean_val_iou



def load_and_evaluate(model_path):
    set_device()
    # Load the saved model state dictionary
    # model = UNET(in_channels = 3, out_channels=3)
    model = torch.load(model_path, map_location=torch.device(DEVICE)).to(DEVICE)
    _, val_loader = data_loaders()
    loss_func = nn.CrossEntropyLoss(weight = LOSS_WEIGHT) # will more epochs help or will more skewed weights help
    mean_val_loss, mean_val_iou = evaluate_validation(model, val_loader, DEVICE, loss_func)
    print(f"Val Mean Loss: {mean_val_loss}, Validation IoU: {mean_val_iou} for model at {model_path}")


    model = torch.load(model_path, map_location=torch.device(DEVICE)).to(DEVICE)
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <model_name>")
        sys.exit(1)

    if os.path.exists(os.path.join("models", sys.argv[1])):
        print("Model name already exists. Please choose a different name.")
        sys.exit(1)

    main(sys.argv[1])