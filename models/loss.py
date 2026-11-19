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
    u_t  = F.conv2d(u,  deriv_t, padding=(1,0)) 
    u_x  = F.conv2d(u,  deriv_x, padding=(0,1))   
    u_xx = F.conv2d(u_x, deriv_x, padding=(0,1))  

    r = (u_t + u * u_x - coef * u_xx).abs()  

    return r

def darcy_visualization(x1_hat, a, device=torch.device('cuda')):
    device = x1_hat.device
    dtype  = x1_hat.dtype
    u = x1_hat[:, 0:1, :, :]
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 1, 3) / (2 * (1/128))
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=dtype, device=device).view(1, 1, 3, 1) / (2 * (1/128))
    grad_x_next_x = F.conv2d(u, deriv_x, padding=(0, 1))
    grad_x_next_y = F.conv2d(u, deriv_y, padding=(1, 0))
    grad_x_next_x = a * grad_x_next_x
    grad_x_next_y = a * grad_x_next_y
    result = F.conv2d(grad_x_next_x, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y, deriv_y, padding=(1, 0))
    residual = result + 1
    
    return residual.abs()

def kol_visualization(w, re=1000.0, dt=1/32):
    nx = w.size(2)
    device = w.device
    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy
    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)
    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:-1]) - f
    
    return torch.abs(residual)

def masked_mse(pred, target, mask):
    diff = (pred - target).abs() * mask
    
    return diff.sum() / mask.sum()

def phys_loss_burgers(u, coef, t0=0.0, t1=1.0, x0=0.0, x1=1.0):
    
    device = u.device
    dtype  = u.dtype
    B, C, H, W = u.shape
    dt = (t1 - t0) / (H)
    dx = (x1 - x0) / (W)
    deriv_t  = torch.tensor([[-1],[0],[1]], device=device, dtype=dtype).view(1,1,3,1) / (2 * dt)
    deriv_x  = torch.tensor([[-1,0,1]],     device=device, dtype=dtype).view(1,1,1,3) / (2 * dx)

    u_t  = F.conv2d(u,  deriv_t,  padding=(1,0))
    u_x  = F.conv2d(u,  deriv_x,  padding=(0,1))
    u_xx = F.conv2d(u_x,  deriv_x, padding=(0,1))
    r = (u_t + u * u_x - coef * u_xx)[:, 0, 1:-1, 2:-2]
    
    return (r.abs() ** 2).mean()

def phys_loss_kol(w, re=1000.0, dt=1/32):

    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device
    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy
    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)
    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:-1]) - f

    return (residual**2).mean()

def phys_loss_darcy(x1_hat, a, device=torch.device('cuda')):
    device = x1_hat.device
    dtype  = x1_hat.dtype
    u = x1_hat[:, 0:1, :, :]
    a = a.to('cuda')
    
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 1, 3) / (2 * (1/128))
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=dtype, device=device).view(1, 1, 3, 1) / (2 * (1/128))
    grad_x_next_x = F.conv2d(u, deriv_x, padding=(0, 1))
    grad_x_next_y = F.conv2d(u, deriv_y, padding=(1, 0))
    grad_x_next_x = a * grad_x_next_x
    grad_x_next_y = a * grad_x_next_y
    result = F.conv2d(grad_x_next_x, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y, deriv_y, padding=(1, 0))
    residual = result + 1
    residual = residual[:,:,2:-2, 2:-2]
    residual = residual.squeeze()
    
    return (residual**2).mean()

def phys_loss_rd(u, nu, rho, t0=0.0, t1=1.0, x0=0.0, x1=1.0):

    device = u.device
    dtype  = u.dtype
    B, C, H, W = u.shape

    # dt = (t1 - t0) / (H - 1)
    # dx = (x1 - x0) / (W - 1)
    dt = 1/ 101
    dx = 16.0 / (1024 - 1)

    deriv_t = torch.tensor([[-1],[0],[1]], device=device, dtype=dtype)\
                .view(1,1,3,1) / (2 * dt)

    deriv_x  = torch.tensor([[-1,0,1]], device=device, dtype=dtype)\
                .view(1,1,1,3) / (2 * dx)

    deriv_xx = torch.tensor([[1,-2,1]], device=device, dtype=dtype)\
                .view(1,1,1,3) / (dx * dx)

    u_t  = F.conv2d(u, deriv_t, padding=(1,0))
    u_xx = F.conv2d(u, deriv_xx, padding=(0,1))

    reaction = rho * u * (1 - u)
    r = (u_t - nu * u_xx - reaction)[:, 0, 1:-1, 2:-2]

    return (r.abs() ** 2).mean()

def _parse_gray_scott_coef(coef):
    if isinstance(coef, dict):
        delta_a = coef["delta_a"]
        delta_b = coef["delta_b"]
        feed    = coef["feed"]
        kill    = coef["kill"]
    elif isinstance(coef, (tuple, list)) and len(coef) == 4:
        delta_a, delta_b, feed, kill = coef
    else:
        raise ValueError(
            "coef for gray_scott must be dict "
            "({'delta_a','delta_b','feed','kill'}) or tuple/list of length 4"
        )
    return delta_a, delta_b, feed, kill

def phys_loss_gray(
    u,
    coef,
    t0=0.0,
    t1=1.0,
    x0=-1.0,
    x1=1.0,
    y0=-1.0,
    y1=1.0,
):
    if u.ndim != 5:
        raise ValueError(f"gray_scott input must have shape [B, 2, T, H, W], got {u.shape}")
    if u.size(1) != 2:
        raise ValueError(f"gray_scott expects 2 channels [A, B], got C={u.size(1)}")

    device = u.device
    dtype = u.dtype
    Bsz, C, T, H, W = u.shape

    if isinstance(coef, dict) and "t1" in coef:
        t1 = coef["t1"]

    delta_a, delta_b, feed, kill = _parse_gray_scott_coef(coef)

    delta_a = torch.as_tensor(delta_a, device=device, dtype=dtype)
    delta_b = torch.as_tensor(delta_b, device=device, dtype=dtype)
    feed    = torch.as_tensor(feed,    device=device, dtype=dtype)
    kill    = torch.as_tensor(kill,    device=device, dtype=dtype)

    dt = (t1 - t0) / T
    dx = (x1 - x0) / W
    dy = (y1 - y0) / H

    A = u[:, 0:1, :, :, :]
    Bv = u[:, 1:2, :, :, :]

    deriv_t = torch.tensor(
        [[[-1.0]], [[0.0]], [[1.0]]], device=device, dtype=dtype
    ).view(1, 1, 3, 1, 1) / (2.0 * dt)

    deriv_xx = torch.tensor(
        [[[1.0, -2.0, 1.0]]], device=device, dtype=dtype
    ).view(1, 1, 1, 1, 3) / (dx * dx)

    deriv_yy = torch.tensor(
        [[[1.0], [-2.0], [1.0]]], device=device, dtype=dtype
    ).view(1, 1, 1, 3, 1) / (dy * dy)

    A_t = F.conv3d(A,  deriv_t,  padding=(1, 0, 0))
    B_t = F.conv3d(Bv, deriv_t,  padding=(1, 0, 0))

    A_xx = F.conv3d(A,  deriv_xx, padding=(0, 0, 1))
    A_yy = F.conv3d(A,  deriv_yy, padding=(0, 1, 0))
    B_xx = F.conv3d(Bv, deriv_xx, padding=(0, 0, 1))
    B_yy = F.conv3d(Bv, deriv_yy, padding=(0, 1, 0))

    lap_A = A_xx + A_yy
    lap_B = B_xx + B_yy

    reaction = A * (Bv ** 2)

    r_A = A_t - delta_a * lap_A + reaction - feed * (1.0 - A)
    r_B = B_t - delta_b * lap_B - reaction + (feed + kill) * Bv

    r_A = r_A[:, :, 1:-1, 1:-1, 1:-1]
    r_B = r_B[:, :, 1:-1, 1:-1, 1:-1]

    return (r_A.square().mean() + r_B.square().mean()) / 2.0

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

            return phys_loss_burgers(predict, coef=coef)

        elif self.type == 'kolmo':

            return phys_loss_kol(predict, re=coef)

        elif self.type == 'darcy':

            return phys_loss_darcy(predict, a=coef)
        
        elif self.type == "reaction_diffusion":
            return phys_loss_rd(predict, nu=coef['nu'], rho=coef['rho'])
        
        elif self.type == "gray":
            return phys_loss_gray(predict, coef=coef)
