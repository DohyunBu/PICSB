import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def burgers_visualization(u, coef):
    device = u.device
    dtype  = u.dtype
    dx = 1/128
    dt = 1/128

    deriv_t = torch.tensor([[-1],[0],[1]], device=device, dtype=dtype).view(1,1,3,1) / (2 * dt)
    deriv_x = torch.tensor([[-1,0,1]],     device=device, dtype=dtype).view(1,1,1,3) / (2 * dx)
    # deriv_xx = torch.tensor([[1,-2,1]],     device=device, dtype=dtype).view(1,1,1,3) / (dx * dx)
    u_t  = F.conv2d(u,  deriv_t, padding=(1,0)) 
    u_x  = F.conv2d(u,  deriv_x, padding=(0,1))   
    u_xx = F.conv2d(u_x, deriv_x, padding=(0,1))  

    r = (u_t + u * u_x - coef * u_xx).abs()  

    return r

def kol_visualization(w, re=1000.0, dt=1/32):
    # w [b t h w]
    nx = w.size(2)
    device = w.device
    # 시간 축 양 끝 프레임빼고 푸리에 변환. 푸리에 변환 전에는 이미지의 픽셀이 곧 해당 좌표에서 갖는 물리량이었는데, 변환 이후에는 특정 주파수에 해당하는 푸리에 계수가 됨
    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    
    # 이게 곧 푸리에 공간에선 미분이라카이..
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy
    # wt는 인접 픽셀 (시간 축)과의 차이
    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)

    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:-1]) - f
    # residual_loss = (residual**2).mean()
    return torch.abs(residual)

def masked_mse(pred, target, mask):
    diff = (pred - target).abs() * mask
    
    return diff.sum() / mask.sum()

def burgers_residual(u, coef, t0=0.0, t1=1.0, x0=0.0, x1=1.0):
    
    device = u.device
    dtype  = u.dtype
    B, C, H, W = u.shape
    dt = (t1 - t0) / (H)
    dx = (x1 - x0) / (W)
    deriv_t  = torch.tensor([[-1],[0],[1]], device=device, dtype=dtype).view(1,1,3,1) / (2 * dt)
    deriv_x  = torch.tensor([[-1,0,1]],     device=device, dtype=dtype).view(1,1,1,3) / (2 * dx)
    # deriv_xx = torch.tensor([[1,-2,1]],     device=device, dtype=dtype).view(1,1,1,3) / (dx * dx)

    u_t  = F.conv2d(u,  deriv_t,  padding=(1,0))
    u_x  = F.conv2d(u,  deriv_x,  padding=(0,1))
    u_xx = F.conv2d(u_x,  deriv_x, padding=(0,1))

    r = (u_t + u * u_x - coef * u_xx)[:, 0, 1:-1, 2:-2]
    return (r.abs() ** 2).mean()

def phys_loss_kol(w, re=1000.0, dt=1/32):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device
    # 시간 축 양 끝 프레임빼고 푸리에 변환. 푸리에 변환 전에는 이미지의 픽셀이 곧 해당 좌표에서 갖는 물리량이었는데, 변환 이후에는 특정 주파수에 해당하는 푸리에 계수가 됨
    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    
    # 이게 곧 푸리에 공간에선 미분이라카이..
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy
    # wt는 인접 픽셀 (시간 축)과의 차이
    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)

    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:-1]) - f
    # residual_loss = (residual**2).mean()
    return (residual**2).mean()


class PhysLoss(nn.Module):
    def __init__(self, pde_type: str, loss_type: str = "l2"):
        super().__init__()
        self.type = pde_type.lower()

        if loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type == "l2":
            self.loss_fn = F.mse_loss

    def forward(self, predict: torch.Tensor, *, coef=None):
        if self.type == "burgers":

            return burgers_residual(predict, coef=coef)

        elif self.type == 'kolmo':

            return phys_loss_kol(predict, re=coef)
