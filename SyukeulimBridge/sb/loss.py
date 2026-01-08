import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# def burgers_pde_mse(u, coef):
#     device = u.device
#     dtype  = u.dtype
#     dx = 1/128
#     dt = 1/128
#     deriv_t = torch.tensor([[-1],[0],[1]], device=device, dtype=dtype).view(1,1,3,1) / (2 * dt)
#     deriv_x = torch.tensor([[-1,0,1]],     device=device, dtype=dtype).view(1,1,1,3) / (2 * dx)
#     deriv_xx = torch.tensor([[1,-2,1]],     device=device, dtype=dtype).view(1,1,1,3) / (dx * dx)
    
#     u_t  = F.conv2d(u,  deriv_t, padding=(1,0))
#     u_x  = F.conv2d(u,  deriv_x, padding=(0,1))
#     u_xx = F.conv2d(u, deriv_xx, padding=(0,1))

#     r = (u_t + u * u_x - coef * u_xx)[:, 0, 1:-1, 1:-1]  
#     return ((r).abs()**2).mean()

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

    r = (u_t + u * u_x - coef * u_xx)[:, 0, 1:-1, 1:-1]   
    r = (r).abs() * mask[:, 0, 1:-1, 1:-1]  
    return r.sum() / ((mask[:, 0, 1:-1, 1:-1]).sum())

def pde_loss_visualization(u, coef):
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

    r = (u_t + u * u_x - coef * u_xx).abs()  

    return r

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


def voriticity_residual(w, re=1000.0, t0=0.0, t1=1.0, x0=0.0, x1=2*np.pi, y0=0.0, y1=2*np.pi):

    w = w.clone()
    with torch.enable_grad():
        w.requires_grad_(True)
        device = w.device
        dtype = w.dtype
        B, T, H, W = w.shape
        dt = (t1 - t0) / (T)
        Lx = (x1 - x0)
        Ly = (y1 - y0)
        dx = Lx / (H)
        dy = Ly / (W)
        w_mid = w[:, 1:-1]
        w_h = torch.fft.fft2(w_mid, dim=(2, 3))

        # physical wavenumbers (radians per unit length)
        k_x = (2 * np.pi) * torch.fft.fftfreq(H, d=dx, device=device, dtype=dtype)  # (H,)
        k_y = (2 * np.pi) * torch.fft.fftfreq(W, d=dy, device=device, dtype=dtype)
        k_x = k_x.view(1, 1, H, 1)
        k_y = k_y.view(1, 1, 1, W)
        # Negative Laplacian in Fourier space
        lap = (k_x ** 2 + k_y ** 2).clone()
        lap[..., 0, 0] = 1.0
        psi_h = w_h / lap

        u_h = 1j * k_y * psi_h
        v_h = -1j * k_x * psi_h
        wx_h = 1j * k_x * w_h
        wy_h = 1j * k_y * w_h
        wlap_h = -lap * w_h

        u    = torch.fft.ifft2(u_h,    dim=(2, 3)).real
        v    = torch.fft.ifft2(v_h,    dim=(2, 3)).real
        wx   = torch.fft.ifft2(wx_h,   dim=(2, 3)).real
        wy   = torch.fft.ifft2(wy_h,   dim=(2, 3)).real
        wlap = torch.fft.ifft2(wlap_h, dim=(2, 3)).real
        advection = u*wx + v*wy

        wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

        # establish forcing term
        x = torch.linspace(x0, x1, H, device=device, dtype=dtype)
        y = torch.linspace(y0, y1, W, device=device, dtype=dtype)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        f = -4*torch.cos(4*Y)

        residual = wt + (advection - (1.0 / re) * wlap + 0.1*w_mid) - f
        
    return (residual.abs() ** 2).mean()


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

            return voriticity_residual(predict, re=coef)
