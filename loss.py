import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def burgers_pde_mse(u, coef):
    device = u.device
    dtype  = u.dtype
    dx = 1/128
    dt = 1/128
    deriv_t = torch.tensor([[-1],[0],[1]], device=device, dtype=dtype).view(1,1,3,1) / (2 * dt)
    deriv_x = torch.tensor([[-1,0,1]],     device=device, dtype=dtype).view(1,1,1,3) / (2 * dx)
    deriv_xx = torch.tensor([[1,-2,1]],     device=device, dtype=dtype).view(1,1,1,3) / (dx * dx)
    
    u_t  = F.conv2d(u,  deriv_t, padding=(1,0))
    u_x  = F.conv2d(u,  deriv_x, padding=(0,1))
    u_xx = F.conv2d(u, deriv_xx, padding=(0,1))

    # coef = torch.as_tensor(coef, device=device, dtype=dtype)
    r = (u_t + u * u_x - coef * u_xx)[:, 0, 1:-1, 1:-1]  
    return ((r).abs()**2).mean()

def burgers_pde_mse_observed_pixel(u, coef, mask):
    device = u.device
    dtype  = u.dtype

    dx = 1/128
    dt = 1/128
    deriv_t = torch.tensor([[-1],[0],[1]], device=device, dtype=dtype).view(1,1,3,1) / (2 * dt)
    deriv_x = torch.tensor([[-1,0,1]],     device=device, dtype=dtype).view(1,1,1,3) / (2 * dx)
    deriv_xx = torch.tensor([[1,-2,1]],     device=device, dtype=dtype).view(1,1,1,3) / (dx * dx)
    u_t  = F.conv2d(u,  deriv_t, padding=(1,0))
    u_x  = F.conv2d(u,  deriv_x, padding=(0,1))
    u_xx = F.conv2d(u, deriv_xx, padding=(0,1))

    # coef = torch.as_tensor(coef, device=device, dtype=dtype)
    r = (u_t + u * u_x - coef * u_xx)[:, 0, 1:-1, 1:-1]   
    r = (r).abs() * mask[:, 0, 1:-1, 1:-1]  
    return r.sum() / ((mask[:, 0, 1:-1, 1:-1]).sum())

def burgers_pde_all_pixel(u, coef):
    device = u.device
    dtype  = u.dtype
    dx = 1/128
    dt = 1/128

    deriv_t = torch.tensor([[-1],[0],[1]], device=device, dtype=dtype).view(1,1,3,1) / (2 * dt)
    deriv_x = torch.tensor([[-1,0,1]],     device=device, dtype=dtype).view(1,1,1,3) / (2 * dx)
    deriv_xx = torch.tensor([[1,-2,1]],     device=device, dtype=dtype).view(1,1,1,3) / (dx * dx)
    u_t  = F.conv2d(u,  deriv_t, padding=(1,0)) 
    u_x  = F.conv2d(u,  deriv_x, padding=(0,1))   
    u_xx = F.conv2d(u, deriv_xx, padding=(0,1))  

    # coef = torch.as_tensor(coef, device=device, dtype=dtype)

    r = (u_t + u * u_x - coef * u_xx).abs()  

    return r

def masked_mse(pred, target, mask):
    diff = (pred - target).abs() * mask
    
    return diff.sum() / mask.sum()
