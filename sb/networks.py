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
        # x: [batch_size, feature_channels, height, width]
        # t: [batch_size, condition_dim]
        h = self.fc(t)  # [batch_size, feature_channels * 2]
        h = h.view(h.size(0), h.size(1), 1, 1)  # [batch_size, feature_channels * 2, 1, 1]
        gamma, beta = torch.chunk(h, 2, dim=1)  # 分成gamma和beta
        x = self.norm(x)  # 实例归一化
        x = gamma * x + beta  # 应用条件归一化
        return x

# 定义两层UNet
# class SimpleUNet(nn.Module):
#     def __init__(self, condition_dim=128):
#         super(SimpleUNet, self).__init__()
#         # 编码器
#         self.encoder_conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
#         self.encoder_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

#         # AdaIN层
#         self.adain1 = AdaIN(128, condition_dim)
#         self.adain2 = AdaIN(256, condition_dim)

#         # 解码器
#         self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 上采样
#         self.decoder_conv1 = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)  # 解码器卷积
#         self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # 上采样
#         self.decoder_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 解码器卷积
#         self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)  # 输出层

#     def forward(self, x_vector, t):
#         x = x_vector.reshape(x_vector.shape[0], 1, 128, 128)
#         x1 = F.leaky_relu(self.encoder_conv1(x))  # [batch_size, 32, 32, 32]
#         x1 = self.adain1(x1, t)  # 应用AdaIN
#         x2 = self.pool(x1)  # [batch_size, 32, 16, 16]

#         x2 = F.leaky_relu(self.encoder_conv2(x2))  # [batch_size, 64, 16, 16]
#         x2 = self.adain2(x2, t)  # 应用AdaIN
#         x3 = self.pool(x2)  # [batch_size, 64, 8, 8]
    
#         # 解码器部分
#         x4 = self.upconv1(x3)  # [batch_size, 32, 16, 16]
#         x4 = torch.cat([x4, x2], dim=1)  # 跳跃连接 [batch_size, 64, 16, 16]
#         x4 = F.leaky_relu(self.decoder_conv1(x4))  # [batch_size, 32, 16, 16]

#         x5 = self.upconv2(x4)  # [batch_size, 32, 32, 32]
#         x5 = torch.cat([x5, x1], dim=1)  # 跳跃连接 [batch_size, 64, 32, 32]
#         x5 = F.leaky_relu(self.decoder_conv2(x5))  # [batch_size, 32, 32, 32]
#         output = self.final_conv(x5)
#         return output.reshape(x_vector.shape[0], -1)
    
class SimpleUNet(nn.Module):
    def __init__(self, condition_dim=128):
        super(SimpleUNet, self).__init__()
        self.encoder_conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.adain1 = AdaIN(128, condition_dim)
        self.adain2 = AdaIN(256, condition_dim)

        # --- 문제 지점 최소 수정: ConvTranspose2d(stride=2) 제거 ---
        # upsample은 interpolate로 하되, 채널만 맞추기 위한 conv로 대체
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
        x4 = self.upconv1(x4)                 # (B,256,64,64) -> (B,128,64,64)
        x4 = torch.cat([x4, x2], dim=1)       # (B,128+256,64,64) = (B,384,64,64)
        x4 = F.leaky_relu(self.decoder_conv1(x4))

        x5 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = self.upconv2(x5)                 # (B,128,128,128) -> (B,128,128,128)
        x5 = torch.cat([x5, x1], dim=1)       # (B,128+128,128,128) = (B,256,128,128)
        x5 = F.leaky_relu(self.decoder_conv2(x5))

        output = self.final_conv(x5)
        return output.reshape(x_vector.shape[0], -1)


# class SimpleUNet(nn.Module):
#     def __init__(self, condition_dim=128):
#         super(SimpleUNet, self).__init__()
#         # 编码器
#         self.encoder_conv1 = nn.Conv2d(1, 256, kernel_size=3, padding=1)
#         self.encoder_conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.encoder_conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

#         # AdaIN层
#         self.adain1 = AdaIN(256, condition_dim)
#         self.adain2 = AdaIN(512, condition_dim)
#         self.adain3 = AdaIN(1024, condition_dim)
#         # 解码器
#         self.upconv0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) 
#         self.decoder_conv0 = nn.Conv2d(1024+512, 512, kernel_size=3, padding=1) 
#         self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) 
#         self.decoder_conv1 = nn.Conv2d(512+256, 256, kernel_size=3, padding=1) 
#         self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
#         self.decoder_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  
#         self.final_conv = nn.Conv2d(256, 1, kernel_size=3, padding=1) 

#     def forward(self, x_vector, t):
#         x = x_vector.reshape(x_vector.shape[0], 1, 128, 128)
#         x1 = F.leaky_relu(self.encoder_conv1(x))  # [batch_size, 32, 32, 32]
#         x1 = self.adain1(x1, t)  # 应用AdaIN
#         x2 = self.pool(x1)  # [batch_size, 32, 16, 16]

#         x2 = F.leaky_relu(self.encoder_conv2(x2))  # [batch_size, 64, 16, 16]
#         x2 = self.adain2(x2, t)  # 应用AdaIN
#         x3 = self.pool(x2)  # [batch_size, 64, 8, 8]

#         x3 = F.leaky_relu(self.encoder_conv3(x3))  # [batch_size, 64, 16, 16]
#         x3 = self.adain3(x3, t)  # 应用AdaIN
#         x4 = self.pool(x3)
        
#         # 解码器部分
#         x5 = self.upconv0(x4)  # [batch_size, 32, 16, 16]
#         x5 = torch.cat([x5, x3], dim=1)  # 跳跃连接 [batch_size, 64, 16, 16]
#         x5 = F.leaky_relu(self.decoder_conv0(x5))  # [batch_size, 32, 16, 16]
        
#         x6 = self.upconv1(x5)  # [batch_size, 32, 16, 16]
#         x6 = torch.cat([x6, x2], dim=1)  # 跳跃连接 [batch_size, 64, 16, 16]
#         x6 = F.leaky_relu(self.decoder_conv1(x6))  # [batch_size, 32, 16, 16]

#         x7 = self.upconv2(x6)  # [batch_size, 32, 32, 32]
#         x7 = torch.cat([x7, x1], dim=1)  # 跳跃连接 [batch_size, 64, 32, 32]
#         x7 = F.leaky_relu(self.decoder_conv2(x7))  # [batch_size, 32, 32, 32]
#         output = self.final_conv(x7)
#         return output.reshape(x_vector.shape[0], -1)
    
class scale_model_muti(nn.Module):
    def __init__(self,output_size,hidden_size=16,input_size=2):
        super(scale_model_muti, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
    def forward(self,t,stage):
        B=t.shape[0]
        if torch.is_tensor(stage):
            stage_val = stage.to(device=t.device, dtype=t.dtype).reshape(1, 1)  # (1,1)
        else:
            stage_val = torch.tensor([[float(stage)]], device=t.device, dtype=t.dtype)  # (1,1)

        stage = stage_val.expand(B, 1)  # (B,1), repeat보다 효율적
        x=torch.cat([t,stage],dim=1)
        x=F.leaky_relu(self.fc1(x))
        x=self.fc2(x)
        return x
# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]
    
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
