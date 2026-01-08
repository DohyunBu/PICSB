import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import math
import functools
from .module_util import *

class AdaIN(nn.Module):
    def __init__(self, feature_channels, condition_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(feature_channels, affine=False)
        self.fc = nn.Linear(condition_dim, feature_channels * 2)

    def forward(self, x, cond):
        h = self.fc(cond)                 
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, 2, dim=1)
        x = self.norm(x)
        return gamma * x + beta

class SimpleUNet(nn.Module):
    def __init__(self, condition_dim=128, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder_conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.adain1 = AdaIN(128, condition_dim)
        self.adain2 = AdaIN(256, condition_dim)

        self.upconv1 = nn.Conv2d(256, 128, kernel_size=1)
        self.decoder_conv1 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.decoder_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x, cond):
        x1 = F.leaky_relu(self.encoder_conv1(x))
        x1 = self.adain1(x1, cond)
        x2 = self.pool(x1)

        x2 = F.leaky_relu(self.encoder_conv2(x2))
        x2 = self.adain2(x2, cond)
        x3 = self.pool(x2)

        x4 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = self.upconv1(x4)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = F.leaky_relu(self.decoder_conv1(x4))

        x5 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = self.upconv2(x5)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = F.leaky_relu(self.decoder_conv2(x5))

        out = self.final_conv(x5)  
        return out

class ConditionalUNet(nn.Module):
    def __init__(
        self,
        condition_dim: int = 128,
        in_channels: int = 1,
        out_channels: int = 1,
        nf: int = 64,
        ch_mul=(1, 2, 4, 8, 16),
    ):
        super().__init__()

        self.depth = depth = len(ch_mul) - 1

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_channels, nf, 3)

        time_dim = condition_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            dim_in  = nf * ch_mul[i]
            dim_out = nf * ch_mul[i + 1]

            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Downsample(dim_in, dim_out) if i != (depth - 1) else default_conv(dim_in, dim_out),
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in),
            ]))

        # mid_dim = nf * ch_mul[-1]
        # self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        # self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_channels, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")
        return x

    def forward(self, x, cond):
        B, _, H, W = x.shape
        # cond shape 정리
        if cond.dim() == 4:
            cond = cond.view(cond.size(0), -1)
        elif cond.dim() != 2:
            raise ValueError(f"cond must be (B, condition_dim) or (B, condition_dim, 1, 1), got {cond.shape}")

        x_in = self.check_image_size(x, H, W)
        x0 = self.init_conv(x_in)
        x_skip0 = x0.clone()
        t = cond.to(x0.device)
        h = []
        for b1, b2, downsample in self.downs:
            x0 = b1(x0, t)
            h.append(x0)

            x0 = b2(x0, t)
            h.append(x0)
            x0 = downsample(x0)

        # x0 = self.mid_block1(x0, t)
        # x0 = self.mid_block2(x0, t)

        for b1, b2, upsample in self.ups:
            x0 = torch.cat([x0, h.pop()], dim=1)
            x0 = b1(x0, t)
            x0 = torch.cat([x0, h.pop()], dim=1)
            x0 = b2(x0, t)
            x0 = upsample(x0)

        x0 = torch.cat([x0, x_skip0], dim=1)
        x0 = self.final_res_block(x0, t)
        out = self.final_conv(x0)
        return out


class ConditionalUNet3D(nn.Module):
    def __init__(self, condition_dim=128, in_channels=1, out_channels=1, nf=32, ch_mul=(1,2,4,8)):
        super().__init__()
        self.depth = depth = len(ch_mul) - 1
        block = functools.partial(ResBlock3D, conv=default_conv3d, act=nn.SiLU())

        self.init_conv = default_conv3d(in_channels, nf, kernel_size=7, bias=False)

        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()

        for i in range(depth):
            dim_in  = nf * ch_mul[i]
            dim_out = nf * ch_mul[i+1]

            self.downs.append(nn.ModuleList([
                block(dim_in=dim_in,  dim_out=dim_in,  time_emb_dim=condition_dim),
                block(dim_in=dim_in,  dim_out=dim_in,  time_emb_dim=condition_dim),
                Downsample3d(dim_in, dim_out) if i != (depth-1) else default_conv3d(dim_in, dim_out, 3),
            ]))

            self.ups.insert(0, nn.ModuleList([
                block(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=condition_dim),
                block(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=condition_dim),
                Upsample3d(dim_out, dim_in) if i != 0 else default_conv3d(dim_out, dim_in, 3),
            ]))

        mid_dim = nf * ch_mul[-1]
        self.mid1 = block(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=condition_dim)
        self.mid2 = block(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=condition_dim)

        self.final_res = block(dim_in=nf*2, dim_out=nf, time_emb_dim=condition_dim)
        self.final_conv = nn.Conv3d(nf, out_channels, 3, 1, 1)

    def forward(self, x, cond):
        if cond.dim() == 4:
            cond = cond.view(cond.size(0), -1)
        elif cond.dim() != 2:
            raise ValueError(f"cond must be (B, condition_dim), got {cond.shape}")

        x0 = self.init_conv(x)
        x_skip = x0

        h = []
        for b1, b2, down in self.downs:
            x0 = b1(x0, cond); h.append(x0)
            x0 = b2(x0, cond); h.append(x0)
            x0 = down(x0)

        x0 = self.mid1(x0, cond)
        x0 = self.mid2(x0, cond)

        for b1, b2, up in self.ups:
            x0 = torch.cat([x0, h.pop()], dim=1)
            x0 = b1(x0, cond)
            x0 = torch.cat([x0, h.pop()], dim=1)
            x0 = b2(x0, cond)
            x0 = up(x0)

        x0 = torch.cat([x0, x_skip], dim=1)
        x0 = self.final_res(x0, cond)
        return self.final_conv(x0)
    
class time_embedding(nn.Module):
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

