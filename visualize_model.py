# https://medium.com/@deepeshdeepakdd2/cnn-visualization-techniques-feature-maps-gradient-ascent-aec4f4aaf5bd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from train import data_loaders
from phenobench import PhenoBench
import numpy as np

# Hook function - these listen to a layers input and output, so that when a forward pass is called we will stow away that layers output
def hook_fn(module, input, output):
    global layer_output
    layer_output = output

# Hook setup
# Hook function
def hook_fn(module, input, output):
    global feature_maps
    feature_maps = output

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
def feature_map(model, image, DEVICE):
    # tell the interpreter to find variable layer_output in global scope
    global layer_output
    # Forward pass the image through the model
    model.eval()

    with torch.inference_mode():
        preds = model(image)

    layer_output = layer_output.squeeze()

    return layer_output


# activation maps - what image maximally activates a unit in a layer (uses gradient ascent on a random image to find the image that maximizes a filter's activation)
def activation_map(model, DEVICE, feature_map_idx = 0):
    global feature_maps
    # Input noise
    input_noise = torch.randn(1, 3, 1024, 1024, requires_grad=True, device=DEVICE)
    # Forward pass
    model.eval()
    preds = model(input_noise)

    # Feature maps
    feature_maps = feature_maps.squeeze_().requires_grad_().to(DEVICE)
    

    # Gradient ascent
    steps = 50
    lr = .1
    optimizer = torch.optim.Adam([input_noise], lr=lr)

    for i in range(steps):
        optimizer.zero_grad()
        
        model(input_noise)
        
        loss = -feature_maps[:,feature_map_idx,:,:].mean()
        loss.backward()
        
        optimizer.step()
# integrated gradients - measure the importance of each input feature (pixel value)

# saliency maps - highlight regions of an input image that are the most important

# grad-cam

if __name__ == "__main__":
    
    DATA_PATH = os.path.join("data", "PhenoBench") # OS independent path
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    #train_loader, val_loader = data_loaders()
    model_id = 5265
    model_path = os.path.join('models', str(model_id), 'model.pt')

    # load an image
    train_data = PhenoBench(DATA_PATH, split = "train", target_types=["semantics"])
    image = torch.tensor(np.transpose(np.array(train_data[0]['image']), (2,0,1))).float().unsqueeze(0).to(DEVICE)


    # load the model onto device
    model = torch.load(model_path, map_location=torch.device(DEVICE)).to(DEVICE)
    # recursively search for convolution layers in the model
    

    cnn_layers = {}
    cnn_layers['downs'] = find_conv_layers(model.downs)
    cnn_layers['bottleneck'] = find_conv_layers(model.bottleneck)
    cnn_layers['ups'] = find_conv_layers(model.ups)
    cnn_layers['final_conv'] = [model.final_conv]

    # Register the hook
    handle = cnn_layers['final_conv'][0].register_forward_hook(hook_fn)

    output = feature_map(model, image, DEVICE)
    print(output.shape)
    print(layer_output.shape)

    rows, cols = 1, 3

    fig = plt.figure(figsize=(10, 6))
    # visualize rows * cols number of channels within the feature map
    for i in range(1, (rows * cols) + 1):
        feature_map = layer_output[i-1, :, :].cpu().numpy()
        fig.add_subplot(rows, cols, i)
        plt.imshow(feature_map, cmap='viridis')
        plt.tight_layout()
        plt.axis(False)

    plt.show()


    