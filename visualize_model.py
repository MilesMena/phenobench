# https://medium.com/@deepeshdeepakdd2/cnn-visualization-techniques-feature-maps-gradient-ascent-aec4f4aaf5bd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from train import data_loaders
from phenobench import PhenoBench
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["keymap.quit"] = ['ctrl+w', 'cmd+w', 'q']


# Function to recursively search for Conv2d layers
def find_conv_layers(module):
    conv_layers = []
    for name, sub_module in module.named_children():
        #print(name)
        if isinstance(sub_module, nn.Conv2d) or isinstance(sub_module, nn.ConvTranspose2d):
            conv_layers.append(sub_module)
        else:
            conv_layers.extend(find_conv_layers(sub_module))
    return conv_layers




# feature maps - output of each filter in a given layer for an input image 
def feature_maps( img, cnn_layers, layer_name, layer_idx, model, DEVICE, rows = 3, cols = 3, channel_shift = 0):
    # device-agnosticaly load image as a torch tensor with the proper shape (1,3,1024,1024) and format (float)
    img= torch.tensor(np.transpose(np.array(img), (2,0,1))).float().unsqueeze(0).to(DEVICE)
    
    # Hook function
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output)
    
    layer = cnn_layers[layer_name][layer_idx]
    print(f'CNN filter: {layer}')
    print(f'{layer_name.capitalize()} has {len(cnn_layers[layer_name])} multichannel feature maps')
    
    # register the hook with the layer
    handle = layer.register_forward_hook(hook_fn)
    
    # Forward pass the image through the model
    model.eval()
    with torch.inference_mode():
        preds = model(img)
    
    # Plot the feature maps
    layer_output = feature_maps[0].squeeze()
    print(f'The feature map at index {layer_idx} has shape {layer_output.shape}')

    if rows * cols > layer_output.shape[0]:
        print(f'Feature Map has {layer_output.shape[0]} channels but you tried plotting {rows * cols} channels')
    else:
        fig = plt.figure(figsize=(12, 8))
        for i in range(1, (rows * cols) + 1):
            feature_map = layer_output[channel_shift + i-1, :, :].cpu().numpy()
            fig.add_subplot(rows, cols, i)
            # virdis is purple with lower values and yellow with highest values (purple -> teal -> green -> yellow)
            plt.imshow(feature_map, cmap='viridis')
            plt.title(f'Channel {channel_shift + i}')
            # plt.tight_layout()
            plt.axis(False)
            plt.colorbar()
        fig.suptitle(f"Feature Maps at index {layer_idx} of {layer_name.capitalize()} layer")
        plt.show()

        #plt.gcf().canvas.mpl_connect('key_press_event', close_plot)

        # Add a colorbar
      

    return layer_output.clone()


# activation maps - what image maximally activates a unit in a layer (uses gradient ascent on a random image to find the image that maximizes a filter's activation)
class FM_GA():
    def __init__(self, layers, layer_name, feature_map_idx, model, DEVICE, steps, lr):
        self.model = model
        self.cnn_layers = layers
        self.feature_map_idx = feature_map_idx
        self.layer_name = layer_name
        self.device = DEVICE
        self.steps = steps
        self.lr = lr
        self.activations = []
        
    def hook_fn(self, module, input, output):
        self.activations.append(output)
        
    def gradient_ascent(self):
        # input noise
        input_noise = torch.randn(1, 3, 1024, 1024, requires_grad=True, device=self.device)
        
        # self.model = self.model.to(self.device)
        self.model.eval()
        
        scaler = GradScaler()
        # activate the hook
        layer = self.cnn_layers[self.layer_name][self.feature_map_idx] 
       
        handle = layer.register_forward_hook(self.hook_fn)
        
        # optimizer
        optimizer = torch.optim.Adam([input_noise], lr=self.lr)
        
        # perform gradient ascent
        for i in tqdm(range(self.steps)):
            optimizer.zero_grad()
            
            with autocast():
                self.model(input_noise.clone())
                loss = -self.activations[0][:, self.feature_map_idx, :, :].mean()

            # scaler.scale(loss).backward(retain_graph=True)
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward(retain_graph = True)
            optimizer.step()


        handle.remove()
        return input_noise
    
    def plot(self):
        torch.autograd.set_detect_anomaly(True)
        input_noise = self.gradient_ascent()
        input_noise_display = input_noise.detach().to('cpu').squeeze().permute(1, 2, 0)
        input_noise_display = torch.clamp(input_noise_display, 0, 1)
        return input_noise_display
    
def activation_map(cnn_layers, layer_name, feature_map_idx, model, DEVICE, steps, lr, rows = 1, cols = 1):
    fig = plt.figure(figsize=(12, 8))

    for i in range(rows * cols):
        # layers,  feature_map_idx, layer_name, model, DEVICE, steps, lr
        feature_map_gradient_ascent = FM_GA(cnn_layers,layer_name, i,  model, DEVICE, steps, lr)
        ax = fig.add_subplot(rows, cols, i+1)
        input_noise_display = feature_map_gradient_ascent.plot()
        ax.imshow(input_noise_display)
        ax.axis('off')
        ax.set_title(f"Layer at Index {feature_map_idx}")

        # Add a single colorbar for all the images
    
# integrated gradients - measure the importance of each input feature (pixel value)

# saliency maps - highlight regions of an input image that are the most important

# grad-cam

if __name__ == "__main__":
    
    DATA_PATH = os.path.join("data", "PhenoBench") # OS independent path
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    #train_loader, val_loader = data_loaders()
    model_id = 5265
    model_path = os.path.join('models', str(model_id), 'model.pt')
    # load the model onto device
    model = torch.load(model_path, map_location=torch.device(DEVICE)).to(DEVICE)

    # load an image
    train_data = PhenoBench(DATA_PATH, split = "train", target_types=["semantics"])
    
    # recursively search for convolution layers in the model
    

    cnn_layers = {}
    cnn_layers['downs'] = find_conv_layers(model.downs)
    cnn_layers['bottleneck'] = find_conv_layers(model.bottleneck)
    cnn_layers['ups'] = find_conv_layers(model.ups)
    cnn_layers['final_conv'] = [model.final_conv]
    
    img_idx = 15
    img = train_data[img_idx]['image']
    mask = train_data[img_idx]['semantics']
   
   
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
       
    axs[0].imshow(img)
    axs[1].imshow(mask)
    #img.show()
    
    plt.show(block = False)


    output = feature_maps(img, cnn_layers, 'ups', 7, model, DEVICE,rows = 3, cols = 3, channel_shift= 0)
    #activation_map(cnn_layers, 'downs', 4, model, DEVICE, 50, .1)

    



        