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
from PIL import Image
# DISUSED IMPORTS
# import time
# from dataset import get_batch_idx, custom_collate
# from torchmetrics import JaccardIndex
# from albumentations.pytorch.transforms import ToTensorV2
# import matplotlib.pyplot as plt
# from itertools import islice

# I despise np arrays that display in scientific notaton
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

    
# TODO: Plotting Function(s)
############## Hyperparameters #################
RESIZE = 1024
BATCH_SIZE = 2 # I hear you are supposed to use as much GPU as possible, but batch size affects the loss propagations. Look into this tradeoff 
EPOCHS = 100
LR = .0001
EVALUATE_IN_LOOP = True
TRAIN_PERCENTAGE = 1 # Should nominally be 1, but for testing purposes we can set it to a fraction of the dataset
VAL_PERCENTAGE = 1 # Should nominally be 1, but for testing purposes we can set it to a fraction of the dataset
SAVE_EVERY = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = os.path.join("data", "PhenoBench") # OS independent path
SANDWICH_PATH = os.path.join("data", "sandwich_images")
# calculate weights by processing dataset histogram balancing by class 
LOSS_WEIGHT = ((1/88.45), (1/11.03), (1/.5))         # CLASS LABELS: {0:soil, 1:crop, 2: weed}


def main(model_name):
    # Activate my python environment for this task:                       C:\Users\menam\projects\phenobench\pheno_env\Scripts\activate
    # Install the most stable version of pytorch with GPU configuration:  https://pytorch.org/get-started/locally/
    # use GPU
    set_device()
    
    EPOCHS = int(input("Enter the Epochs would you like to train: "))

    CHANNELS = { # comment out the channels you don't want to include
        "R": 0,
        "G": 1,
        "B": 2,
        # "ExG": 3,
        # "ExR": 4,
        "CIVE": 5,
        # "NDI": 6,
        "YOLO": 7
    }

    train_loader = feature_engineered_data_loaders("train", CHANNELS, TRAIN_PERCENTAGE)
    val_loader = feature_engineered_data_loaders("val", CHANNELS, VAL_PERCENTAGE)
    ################# Init the Model #####################
    #layer = DoubleConv(in_channels=3, out_channels=64).to(DEVICE) # ensure that weights and data are both on GPU
    model = load_model(model_name, CHANNELS)
    #model = UNET(in_channels = len(CHANNELS), out_channels=3).to(device=DEVICE)
    subfolders = [f.name for f in os.scandir("models") if f.is_dir()]
    if model_name in subfolders:
        subfolder_path = os.path.join("models", model_name)
        model_subfolders = [f.name for f in os.scandir(subfolder_path) if f.is_dir()]
        highest_epoch_num_ix = np.argmax(np.array([int(model_sub_dir.split('_')[1]) for model_sub_dir in  model_subfolders]))
        max_epoch = int(model_subfolders[highest_epoch_num_ix].split('_')[1]) + 1
    else: 
        max_epoch = 0

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler() # automated mixed precision. Dynamically scale between float16 and float32 stability and computation increases during back prop
    
    # Loss function. 
    loss_func = nn.CrossEntropyLoss(weight = LOSS_WEIGHT) # will more epochs help or will more skewed weights help
    
    # Define the Adam optimizer.
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
                        "epochs": i + max_epoch,
                        "learning_rate": LR,
                        "loss_weight": LOSS_WEIGHT.tolist(),
                        "resize": RESIZE,
                        "device": DEVICE,
                        "mean_val_loss": mean_val_loss,
                        "mean_val_iou": mean_val_iou,
                        "channels": CHANNELS
                    }

            save_model(model, model_parameters)

        ################## MODEL STATISTICS #############
        # # convert batch loss into np array on the CPU
        # batch_loss = torch.stack(batch_loss).detach().cpu().numpy()
        # epoch_loss.append(batch_loss)
        # if flag:
        #     break

def load_model(model_name, CHANNELS):
    subfolders = [f.name for f in os.scandir("models") if f.is_dir()]
    if model_name in subfolders:
        subfolder_path = os.path.join("models", model_name)
        model_subfolders = [f.name for f in os.scandir(subfolder_path) if f.is_dir()]
        highest_epoch_num_ix = np.argmax(np.array([int(model_sub_dir.split('_')[1]) for model_sub_dir in model_subfolders]))
        print(f'Retraining {model_name} at {model_subfolders[highest_epoch_num_ix]}')
        return torch.load(os.path.join("models",model_name, model_subfolders[highest_epoch_num_ix], "model.pt"), map_location=torch.device(DEVICE)).to(DEVICE)

    else:
        return UNET(in_channels = len(CHANNELS), out_channels=3).to(device=DEVICE)



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



def feature_engineered_data_loaders(split, channels, percentage=1):
    original_data = PhenoBench(DATA_PATH, split = split, target_types=["semantics"])

    new_data = []
    for i in tqdm(range(len(original_data)), colour="yellow", desc=f"Loading {split} data"):

        image = np.load(os.path.join(SANDWICH_PATH, split, original_data[i]['image_name'].split(".")[0] + ".npy"))
        new_image = np.stack([image[:,:,channel] for channel in channels.values()], axis = 2) # select the right channels

        new_data.append({'image': new_image, 'semantics': original_data[i]['semantics'], 'image_name': original_data[i]['image_name']})

    transform = A.Compose([
                A.Resize(height = RESIZE, width = RESIZE)  # original images size:  (1024, 1024)
                ])

    def custom_collate(batch):
        transformed = [transform(image=np.array(item['image']), mask = np.array(item['semantics'])) for item in batch]
        images = [item['image'] for item in transformed]
        masks = [item['mask'] for item in transformed]

        
        return {'images': torch.tensor(np.transpose(np.array(images), (0,3,1,2))).float(), 'masks': torch.tensor(np.array(masks))}
    
    return DataLoader(new_data, batch_size=BATCH_SIZE, collate_fn=custom_collate, drop_last=True, sampler=RandomSampler(new_data, replacement=False, num_samples=int(len(new_data)*percentage)))


def data_loaders(split, percentage=1):
    ############# Init the Phenobench DataLoader #################
    # Phenobench's DataLoader sits on top the directroy and only loads when .__getitem__ is called
    # ex: train_data[image_index]['image']
    data = PhenoBench(DATA_PATH, split = split, target_types=["semantics"])

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

    return DataLoader(data, batch_size=BATCH_SIZE, collate_fn=custom_collate, drop_last=True, sampler=RandomSampler(data, replacement=False, num_samples=int(len(data)*percentage)))


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
    

def load_and_make_submission(model_path):
    set_device()
    model = torch.load(model_path, map_location=torch.device(DEVICE)).to(DEVICE)
    model.eval()

    CHANNELS = { # comment out the channels you don't want to include
        "R": 0,
        "G": 1,
        "B": 2,
        # "ExG": 3,
        # "ExR": 4,
        "CIVE": 5,
        # "NDI": 6,
        "YOLO": 7
    }

    original_data = PhenoBench(DATA_PATH, split = "test")

    new_data = []
    for i in tqdm(range(len(original_data)), colour="yellow", desc=f"Loading test data"):

        image = np.load(os.path.join(SANDWICH_PATH, "test", original_data[i]['image_name'].split(".")[0] + ".npy"))
        new_image = np.stack([image[:,:,channel] for channel in CHANNELS.values()], axis = 2) # select the right channels

        new_data.append({'image': new_image, 'image_name': original_data[i]['image_name']})

    # Save all images predicted from this model to png files in the submission directory
    submission_path = os.path.join("submissions", model_path.split("/")[-2], "semantics")
    if not os.path.exists(submission_path):
        os.makedirs(submission_path)
    # load an image
    for img_idx in range(len(new_data)):
        img = new_data[img_idx]['image']
        pred = model.predict(torch.tensor(np.transpose(np.array(img), (2,0,1))).float().unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
        #convert to ints
        pred = pred.astype(np.uint8)
        image_path = os.path.join(submission_path, new_data[img_idx]['image_name'])
        # save prediciton as an image
        Image.fromarray(pred).save(image_path)
        print(f"Image {img_idx} saved to {image_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <model_name>")
        sys.exit(1)

    if os.path.exists(os.path.join("models", sys.argv[1])):
        retrain = input("Model name already exists. Would you like to continue training? (Y/n): ")
        if retrain.lower() == 'y':
            main(sys.argv[1])
        else:
            sys.exit(1)
    else: 
        main(sys.argv[1])