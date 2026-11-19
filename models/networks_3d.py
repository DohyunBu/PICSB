import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN3D(nn.Module):
    def __init__(self, feature_channels, condition_dim):
        super().__init__()
        self.norm = nn.InstanceNorm3d(feature_channels, affine=False)
        self.fc = nn.Linear(condition_dim, feature_channels * 2)

    def forward(self, x, cond):
        h = self.fc(cond)                     
        h = h.view(h.size(0), h.size(1), 1, 1, 1) 
        gamma, beta = torch.chunk(h, 2, dim=1)  
        x = self.norm(x)
        return gamma * x + beta

class SimpleUNet3D(nn.Module):
    def __init__(self, condition_dim=128, in_channels=4, out_channels=4):
        super().__init__()

        self.encoder_conv1 = nn.Conv3d(in_channels, 128, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.adain1 = AdaIN3D(128, condition_dim)
        self.adain2 = AdaIN3D(256, condition_dim)

        self.upconv1 = nn.Conv3d(256, 128, kernel_size=1)
        self.decoder_conv1 = nn.Conv3d(128 + 256, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.Conv3d(128, 128, kernel_size=1)
        self.decoder_conv2 = nn.Conv3d(128 + 128, 128, kernel_size=3, padding=1)

        self.final_conv = nn.Conv3d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x, cond):
        x1 = F.leaky_relu(self.encoder_conv1(x))   
        x1 = self.adain1(x1, cond)

        x2_in = self.pool(x1)                  

        x2 = F.leaky_relu(self.encoder_conv2(x2_in)) 
        x2 = self.adain2(x2, cond)

        x3 = self.pool(x2)                  

        x4 = F.interpolate(x3, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)                                    
        x4 = self.upconv1(x4)                       
        x4 = torch.cat([x4, x2], dim=1)          
        x4 = F.leaky_relu(self.decoder_conv1(x4))  

        x5 = F.interpolate(x4, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)                                        
        x5 = self.upconv2(x5)                      
        x5 = torch.cat([x5, x1], dim=1)         
        x5 = F.leaky_relu(self.decoder_conv2(x5)) 

        out = self.final_conv(x5)                  

        return out


class TimeEmbedding(nn.Module):
    def __init__(self, output_dim, hidden_dim=16, input_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, t, stage):
        B = t.shape[0]
        if torch.is_tensor(stage):
            stage_val = stage.to(device=t.device, dtype=t.dtype).reshape(1, 1)
        else:
            stage_val = torch.tensor([[float(stage)]], device=t.device, dtype=t.dtype)

        stage_feat = stage_val.expand(B, 1)
        x = torch.cat([t, stage_feat], dim=1) 
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)            