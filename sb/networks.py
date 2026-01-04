import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class AdaIN(nn.Module):
    def __init__(self, feature_channels, condition_dim):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(feature_channels, affine=False)
        self.fc = nn.Linear(condition_dim, feature_channels * 2)

    def forward(self, x, t):
        h = self.fc(t)  
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, 2, dim=1)  
        x = self.norm(x) 
        x = gamma * x + beta 
        return x
    
class SimpleUNet(nn.Module):
    def __init__(self, condition_dim=128):
        super(SimpleUNet, self).__init__()
        self.encoder_conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
                
        self.adain1 = AdaIN(128, condition_dim)
        self.adain2 = AdaIN(256, condition_dim)

        self.upconv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.decoder_conv1 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.decoder_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x_vector, t):
        x = x_vector.reshape(x_vector.shape[0], 1, 128, 128)
        x1 = F.leaky_relu(self.encoder_conv1(x))
        x1 = self.adain1(x1, t)
        x2 = self.pool(x1)

        x2 = F.leaky_relu(self.encoder_conv2(x2))
        x2 = self.adain2(x2, t)
        x3 = self.pool(x2)

        x4 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = self.upconv1(x4)               
        x4 = torch.cat([x4, x2], dim=1)      
        x4 = F.leaky_relu(self.decoder_conv1(x4))

        x5 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = self.upconv2(x5)         
        x5 = torch.cat([x5, x1], dim=1)     
        x5 = F.leaky_relu(self.decoder_conv2(x5))

        output = self.final_conv(x5)
        return output.reshape(x_vector.shape[0], -1)

    
class scale_model_muti(nn.Module):
    def __init__(self,output_size,hidden_size=16,input_size=2):
        super(scale_model_muti, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
    def forward(self,t,stage):
        B=t.shape[0]
        if torch.is_tensor(stage):
            stage_val = stage.to(device=t.device, dtype=t.dtype).reshape(1, 1)
        else:
            stage_val = torch.tensor([[float(stage)]], device=t.device, dtype=t.dtype) 

        stage = stage_val.expand(B, 1) 
        x=torch.cat([t,stage],dim=1)
        x=F.leaky_relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
class CustomDataset(Dataset):
    def __init__(self, data, mask=None):
        self.data = data
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mask is None:
            return self.data[idx]
        return self.data[idx], self.mask[idx]
