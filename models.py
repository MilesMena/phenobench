import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


############ UNET ###############
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), # bias is false since we batchnorm
            nn.BatchNorm2d(out_channels), # batch norm for better gradients
            nn.ReLU(inplace=True),
            # doulbe conv
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False), # kernal size, stride, padding
            nn.BatchNorm2d(out_channels), # batch norm for better gradients
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, features = [64,128,256,512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.ups = nn.ModuleList()
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size = 2, stride = 2)
            )

            self.ups.append(DoubleConv(feature * 2, feature))

        

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)
        self.softmax = nn.Softmax(dim = 1 )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] 

        for idx in range(0, len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx + 1](concat_skip)

        
        return self.final_conv(x) # we need raw logits to calculate a proper loss
    
    def predict(self, x):
        x = self.forward(x)
        x = self.softmax(x)
        return torch.argmax(x, dim = 1)
    
    def predict_from_logits(self, x):
        x = self.softmax(x)
        return torch.argmax(x, dim = 1)

####### UNET with CBAM ################
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // 16, bias=False),
                               nn.ReLU(),
                               nn.Linear(in_channels // 16, in_channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # pool with both the average and max, pass it through the MLP, take the sigmoid
        # channel_attetion_weights example: tensor([[[[1], [0], [.5]]]]) -> pay full attention to the first layer, disregard the second layer, half the values of the third layer
        return self.sigmoid(self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1)) + self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2,1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # concat the av and max the apply a conv2d over it       
        return self.sigmoid(self.conv(torch.cat([torch.mean(x, dim=1, keepdim=True), max_out], dim=1))) * x

class UNET_CBAM(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, features = [64,128,256,512]):
        super(UNET_CBAM, self).__init__()
        self.downs = nn.ModuleList()
        # CBAM module
        self.bottleneck = nn.Sequential(DoubleConv(features[-1], features[-1] * 2),
                                        ChannelAttention(features[-1] * 2),
                                        SpatialAttention()
                                    )
                                        
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # We implement CBAM here
        for feature in features:
            self.downs.append(nn.Sequential(
                DoubleConv(in_channels, feature),
                ChannelAttention(feature),
                SpatialAttention()
                )
            )
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size = 2, stride = 2)
            )

            # CBAM Module
            self.ups.append(nn.Sequential(DoubleConv(feature * 2, feature),
                            ChannelAttention(feature),
                            SpatialAttention()
                            )
                            )
                            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)
        self.softmax = nn.Softmax(dim = 1 )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] 

        for idx in range(0, len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx + 1](concat_skip)

        
        return self.final_conv(x) # we need raw logits to calculate a proper loss
    
    def predict(self, x):
        x = self.forward(x)
        x = self.softmax(x)
        return torch.argmax(x, dim = 1)
    
    def predict_from_logits(self, x):
        x = self.softmax(x)
        return torch.argmax(x, dim = 1)




