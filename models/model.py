import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import copy
import os
import torch

from models.visualize import *
import models.networks
import models.networks_3d
from models.loss import *

def _to_device_float(x, device):
    if hasattr(x, "to"):
        return x.to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.as_tensor(x, device=device, dtype=torch.float32)

def _to_device_bool(m, device):
    if m is None:
        return None
    if hasattr(m, "to"):
        return m.to(device=device, dtype=torch.bool, non_blocking=True)
    return torch.as_tensor(m, device=device, dtype=torch.bool)

def _prepare_unet_input(low_data, high_data, mask_high, device):
    low_data  = _to_device_float(low_data, device)
    high_data = _to_device_float(high_data, device)
    mask_high = _to_device_bool(mask_high, device)

    if low_data.ndim == 5:
        x1_fixed = (high_data * mask_high).detach()
        low_data = torch.where(mask_high, x1_fixed, low_data)
        high_data = torch.where(mask_high, x1_fixed, high_data)
        return low_data, high_data, mask_high, "3d"
    
    B, C, H, W = low_data.shape

    if C == 1:
        x1_fixed = (high_data * mask_high).detach()
        low_data  = torch.where(mask_high, x1_fixed, low_data)
        high_data = torch.where(mask_high, x1_fixed, high_data)
        return low_data, high_data, mask_high, "1d"

    x1_fixed = (high_data * mask_high).detach()
    low_data  = torch.where(mask_high, x1_fixed, low_data)
    high_data = torch.where(mask_high, x1_fixed, high_data)

    return low_data, high_data, mask_high, "2d_time_as_channel"

    
class PICSB(object):
    def __init__(self, train_dataloader, timepoints, val_dataloader=None, time_dim=32, steps=60, base_lr=1e-3,save=False, beta_min=0.01, beta_max=0.01, pde_type='kolmo', in_channels=40):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.best_val_pde = float("inf")
        self.timepoints = timepoints
        self.steps=steps
        self.base_lr=base_lr
        self.delta_t = (timepoints[-1] - timepoints[0]) / self.steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pde_type = pde_type
        self.phys_loss = PhysLoss(pde_type=pde_type, loss_type="l2").to(self.device)
        self.t_list = torch.arange(timepoints[0], timepoints[-1], self.delta_t, device=self.device, dtype=torch.float32)
        self.t_list = torch.cat([self.t_list, torch.tensor([timepoints[-1]], device=self.device, dtype=self.t_list.dtype)])
        self.t_lists_stage=[[] for i in range(len(self.timepoints)-1)]
        for t in self.t_list:
            for t_index,t_point in enumerate(self.timepoints[1:]):
                if t >= t_point:
                    continue
                else:
                    self.t_lists_stage[t_index].append(t)
                    break                
        for stage, t_lst in enumerate(self.t_lists_stage):
            t_lst.append(torch.as_tensor(self.timepoints[stage+1], device=self.device, dtype=torch.float32))
    
        self.in_channels = in_channels
        if self.pde_type == "burgers":
            self.v_fore = models.networks.SimpleUNet(
                condition_dim=time_dim,
                in_channels=self.in_channels,
                out_channels=self.in_channels,
            ).to(self.device)
            
        elif self.pde_type == "kolmo":
            self.v_fore = models.networks.SimpleUNet(
                condition_dim=time_dim,
                in_channels=self.in_channels,
                out_channels=self.in_channels,
            ).to(self.device)
            
        elif self.pde_type == "darcy":
            self.v_fore = models.networks.SimpleUNet(
                condition_dim=time_dim,
                in_channels=self.in_channels,
                out_channels=self.in_channels,
            ).to(self.device)
            
        elif self.pde_type == "gray":
            self.v_fore = models.networks_3d.SimpleUNet3D(
                condition_dim=time_dim,
                in_channels=self.in_channels,
                out_channels=self.in_channels,
            ).to(self.device)
        
        elif self.pde_type == "reaction_diffusion":
            self.v_fore = models.networks.SimpleUNet(
                condition_dim=time_dim,
                in_channels=self.in_channels,
                out_channels=self.in_channels,
            ).to(self.device)
            
        self.time_embed = models.networks.time_embedding(output_dim=time_dim).to(self.device)
        decay_params = []
        no_decay_params = []

        for name, param in list(self.v_fore.named_parameters()) + list(self.time_embed.named_parameters()):
            if not param.requires_grad:
                continue
            if name.endswith("bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        self.optimizer_fore = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 1e-2},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.base_lr,
        )     
    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def train(self, n_iterations=10000, model_name=None, known_alpha=None, refresh=100):
        self.loss_history={"PDE_loss_fore":[], "Val_PDE_loss":[]}

        self.v_fore.train()
        self.time_embed.train() 
        loader_it = iter(self.train_dataloader)

        pbar = tqdm(range(n_iterations), desc="Train")
        for iter_idx in pbar:
            val_pde_mean = None
            try:
                batch = next(loader_it)
            except StopIteration:
                loader_it = iter(self.train_dataloader)
                batch = next(loader_it)
            self.optimizer_fore.zero_grad()
            x_0, x_1, mask_high = batch
            if self.pde_type=='darcy':
                a = x_0[:,0:1]
                x_0 = x_0[:,1:2]
                x_1 = x_1[:,1:2]
                mask_high = mask_high[:,1:2]
            x0_fixed, x1_fixed, mask, mode = _prepare_unet_input(x_0, x_1, mask_high, self.device)
            
            # Refresh
            if iter_idx % refresh == 0:
                self.fix_bridge_v_fore = copy.deepcopy(self.v_fore)
                self.fix_bridge_time_embed = copy.deepcopy(self.time_embed)

                self.fix_bridge_v_fore.eval()
                self.fix_bridge_time_embed.eval()

                for p in self.fix_bridge_v_fore.parameters():
                    p.requires_grad_(False)
                for p in self.fix_bridge_time_embed.parameters():
                    p.requires_grad_(False)

            x_1_hat_fixed = self.gen_sur_endpoint(x_0=x0_fixed, x_1_fixed=x1_fixed, mask=mask, v_m=self.fix_bridge_v_fore, t_m=self.fix_bridge_time_embed, one_step=False)
            if self.pde_type=='darcy':
                x_1_hat_fixed = x_1_hat_fixed.abs()

            x_t_fore, t_fore, t_ceil, t_floor = self.Interp_t(x_0=x0_fixed, x_1=x_1_hat_fixed, forward=True)
            x_t_fore = torch.where(mask, x1_fixed, x_t_fore)

            x_1_hat = self.reconstruction(x_0=x_t_fore, x_1_fixed=x1_fixed, mask=mask, v_m=self.v_fore, t_m=self.time_embed, start_t=t_fore, one_step=False)
            if self.pde_type=='darcy':
                x_1_hat = x_1_hat.abs()
            
            if self.pde_type == "kolmo":
                pde_fore = self.phys_loss(x_1_hat, coef=known_alpha)
            elif self.pde_type == "burgers":
                pde_fore = self.phys_loss(x_1_hat, coef=known_alpha)
            elif self.pde_type == "darcy":
                pde_fore = self.phys_loss(x_1_hat, coef=a)
            elif self.pde_type == "reaction_diffusion":
                pde_fore = self.phys_loss(x_1_hat, coef=known_alpha)
            elif self.pde_type == "gray":
                pde_fore = self.phys_loss(x_1_hat, coef=known_alpha)
            
            loss = 1 * pde_fore
            epoch_pde_loss_fore = pde_fore

            loss.backward()
            if self.pde_type == "darcy" or self.pde_type == "reaction_diffusion":
                torch.nn.utils.clip_grad_norm_(self.v_fore.parameters(), max_norm=0.00001)
                torch.nn.utils.clip_grad_norm_(self.time_embed.parameters(), max_norm=0.00001)
            else:
                torch.nn.utils.clip_grad_norm_(self.v_fore.parameters(), max_norm=1)
                torch.nn.utils.clip_grad_norm_(self.time_embed.parameters(), max_norm=1)
            self.optimizer_fore.step()
            do_val = (iter_idx % 100 == 0) or (iter_idx == n_iterations - 1)
            if do_val and self.val_dataloader is not None:
                val_pde_mean = None
                self.v_fore.eval()
                self.time_embed.eval()
                with torch.no_grad():
                    pde_val = 0.0
                    cnt = 0
                    for vb in self.val_dataloader:
                        vx_0, vx_1, vmask_high = vb
                        if self.pde_type=='darcy':
                            va = copy.deepcopy(vx_0[:,0:1])
                            vx_0 = vx_0[:,1:2]
                            vx_1 = vx_1[:,1:2]
                            vmask_high = vmask_high[:,1:2]
                        vx_1_hat = self.eval_fore(x_0=vx_0, x_1=vx_1, mask_high=vmask_high)
                        if self.pde_type == "kolmo":
                            vpde = self.phys_loss(vx_1_hat, coef=known_alpha)
                        elif self.pde_type == "burgers":
                            vpde = self.phys_loss(vx_1_hat, coef=known_alpha)
                        elif self.pde_type == "darcy":
                            vpde = self.phys_loss(vx_1_hat, coef=va)
                        elif self.pde_type == "reaction_diffusion":
                            vpde = self.phys_loss(vx_1_hat, coef=known_alpha)
                        elif self.pde_type == "gray":
                            vpde = self.phys_loss(vx_1_hat, coef=known_alpha)
                        pde_val += float(vpde.detach().cpu().item())
                        cnt += 1
                    val_pde_mean = pde_val / max(cnt, 1)

                    if val_pde_mean < self.best_val_pde:
                        self.best_val_pde = val_pde_mean
                        models_dict = {
                            'v_fore': self.v_fore.state_dict(),
                            'fix_bridge_v_fore': self.fix_bridge_v_fore.state_dict(),
                            'time_embed': self.time_embed.state_dict(),
                            'fix_bridge_time_embed': self.fix_bridge_time_embed.state_dict(),
                            'optimizer_fore': self.optimizer_fore.state_dict(),
                            'best_val_pde': self.best_val_pde,
                            'loss_history': self.loss_history                            
                        }
                        base, ext = os.path.splitext(model_name)
                        model_name_val = f"{base}_val{ext}"
                        torch.save(models_dict, os.path.join(model_name_val))
                self.loss_history['Val_PDE_loss'].append(val_pde_mean)
                self.v_fore.train()
                self.time_embed.train()
            with torch.no_grad():
                self.loss_history['PDE_loss_fore'].append(np.log1p(epoch_pde_loss_fore.item()))
                
            if iter_idx%100==0 and iter_idx!=0:
                torch.cuda.empty_cache()
                models_dict = {
                    'v_fore': self.v_fore.state_dict(),
                    'fix_bridge_v_fore': self.fix_bridge_v_fore.state_dict(),
                    'time_embed': self.time_embed.state_dict(),
                    'fix_bridge_time_embed': self.fix_bridge_time_embed.state_dict(),
                    'optimizer_fore': self.optimizer_fore.state_dict(),
                    'best_val_pde': self.best_val_pde,
                    'loss_history': self.loss_history                            
                }
                torch.save(models_dict, os.path.join(model_name))
            pbar.set_description('processed: %d' % (1 + iter_idx))

            pbar.set_postfix({'PDE_loss':epoch_pde_loss_fore.detach().cpu().numpy(), 'Val_PDE_loss':val_pde_mean})
            pbar.update(1)
    
    # For Training           
    def eval_fore(self, x_0, x_1, mask_high=None):
        self.v_fore.eval()
        self.time_embed.eval()

        with torch.no_grad():
            x0_fix, x1_fix, mask_fix, mod = _prepare_unet_input(x_0, x_1, mask_high, self.device)
            x_last = self.fwd_inference(x0_fix=x0_fix, x1_fix=x1_fix, mask_fix=mask_fix, v_m=self.v_fore, t_m=self.time_embed)

            return x_last
        
    def fwd_inference(self, x0_fix, x1_fix, mask_fix, v_m, t_m, stage_index=0):
        device = self.device
        dtype = torch.float32

        x_t = x0_fix.to(device=device, dtype=dtype)
        x1 = x1_fix.to(device=device, dtype=dtype)
        mask = mask_fix.to(device=device, dtype=torch.bool) if mask_fix is not None else None

        if mask is not None:
            x_t = torch.where(mask, x1, x_t)

        t_list_stage = self.t_lists_stage[stage_index]
        Bp = x_t.shape[0]
        nd = x_t.ndim
        shape = (Bp, 1) + (1,) * (nd - 2)

        for t_new in t_list_stage[:-1]:
            t = t_new.to(device=device, dtype=dtype).expand(Bp, 1)
            cond = t_m(t, 1)
            drift = v_m(x_t, cond) 

            if t_new < t_list_stage[-2]:
                z = torch.randn_like(drift) * (self.delta_t ** 0.5)
                t_norm = (t - 0) / (1 - 0)
                scheduled_noise = self.beta_t(t_norm).view(*shape)
                del_x = drift * self.delta_t + (scheduled_noise ** 0.5) * z
            else:
                del_x = drift * self.delta_t
            x_t = x_t + del_x
            if mask is not None:
                x_t = torch.where(mask, x1, x_t)

        return x_t
    
    # Construct Brownian-bridge
    def Interp_t(self, x_0, x_1, forward=True):
        device = x_0.device
        dtype  = x_0.dtype
        Bp = x_0.shape[0]

        n_stages = len(self.t_lists_stage)
        stage_index = int(torch.randint(low=0, high=n_stages, size=(1,)).item())

        t_list = self.t_lists_stage[stage_index]
        t_list = torch.tensor([float(t) for t in t_list], device=device, dtype=dtype)

        t_floor = t_list[0]
        t_ceil  = t_list[-1]

        if forward:
            candidates = t_list[:-1]
        else:
            candidates = t_list[1:]

        t = candidates[torch.randint(len(candidates), (Bp,), device=device)].reshape(-1, 1)

        Z = torch.randn_like(x_0)
        
        alpha = (t_ceil - t) / (t_ceil - t_floor)  
        beta  = (t - t_floor) / (t_ceil - t_floor)  

        nd = x_0.ndim  
        shape = (Bp, 1) + (1,) * (nd - 2) 

        alpha = alpha.view(*shape)
        beta  = beta.view(*shape)

        scheduled_noise = self.beta_t((t - t_floor) / (t_ceil - t_floor))
        scheduled_noise = scheduled_noise.view(*shape)
        x_1 = x_1.detach()
        x_t = alpha * x_0 + beta * x_1 + torch.sqrt(scheduled_noise * alpha * beta) * Z

        return x_t, t, t_ceil, t_floor
    
    # Generate surrogate endpoint
    def gen_sur_endpoint(
        self, x_0, x_1_fixed, mask, v_m, t_m,
        stage_index: int = 0, start_t=None, one_step=False
    ):
        device = x_0.device
        dtype  = x_0.dtype
        Bp     = x_0.shape[0]

        x_t = x_0.detach()
    
        t_list_stage = self.t_lists_stage[stage_index]
        t_list_stage = torch.as_tensor(
            [float(t) for t in t_list_stage],
            device=device, dtype=dtype
        )
        S = t_list_stage.numel()

        t_floor_stage = t_list_stage[0]
        t_ceil_stage  = t_list_stage[-1]

        if start_t is None:
            start_t = torch.zeros((Bp, 1), device=device, dtype=dtype)
        else:
            if not torch.is_tensor(start_t):
                start_t = torch.tensor(start_t, device=device, dtype=dtype)
            start_t = start_t.to(device=device, dtype=dtype)

            if start_t.ndim == 0:
                start_t = start_t.view(1, 1).expand(Bp, 1)
            elif start_t.ndim == 1:
                start_t = start_t.view(-1, 1)
            if start_t.shape[0] != Bp:
                if start_t.shape[0] == 1:
                    start_t = start_t.expand(Bp, 1)
                else:
                    raise ValueError(f"start_t batch mismatch: got {start_t.shape}, expected ({Bp}, 1)")

        start_idx = torch.searchsorted(t_list_stage, start_t.squeeze(1), right=False) 
        start_idx = start_idx.clamp(0, S - 1)
        
        if one_step:
            t0 = t_list_stage[start_idx].view(Bp, 1)
            y  = t_m(t0, t_ceil_stage)            
            drift = v_m(x_t, y)  
            nd = x_t.ndim
            remaining = (S - 1 - start_idx).to(dtype=dtype).view(Bp, 1, *([1] * (nd - 2)))
            scale = self.delta_t * remaining
            del_x = drift * scale
            x_t = x_t + del_x
            if mask is not None:
                x_t = torch.where(mask, x_1_fixed, x_t)
            return x_t

        max_remain = int((S - 1 - start_idx).max().item())  

        for r in range(max_remain):
            cur_idx = start_idx + r                
            active  = cur_idx < (S - 1)               
            if not torch.any(active):
                break

            act_idx = torch.nonzero(active, as_tuple=False).squeeze(1)

            t_cur = t_list_stage[cur_idx[act_idx]].view(-1, 1) 
            y = t_m(t_cur, t_ceil_stage)
            drift = v_m(x_t[act_idx], y)

            t_norm = (t_cur - t_floor_stage) / (t_ceil_stage - t_floor_stage)
            nd = x_t.ndim
            scheduled_noise = self.beta_t(t_norm).view(-1, 1, *([1] * (nd - 2)))

            z = torch.randn_like(drift) * (self.delta_t ** 0.5)
            if r == max_remain - 1:
                del_x = drift * self.delta_t
            else:
                del_x = drift * self.delta_t + (scheduled_noise ** 0.5) * z

            x_t_act = x_t[act_idx] + del_x
            if mask is not None:
                x_t_act = torch.where(mask[act_idx], x_1_fixed[act_idx], x_t_act)
            x_t = x_t.clone()
            x_t[act_idx] = x_t_act  
            x_t = x_t.detach()

        return x_t
    
    # Reconstruction
    def reconstruction(
        self, x_0, x_1_fixed, mask, v_m, t_m,
        stage_index: int = 0, start_t=None, one_step=False
    ):
        device = x_0.device
        dtype  = x_0.dtype
        Bp     = x_0.shape[0]
    
        x_t = x_0
        
        t_list_stage = self.t_lists_stage[stage_index]
        t_list_stage = torch.as_tensor(
            [float(t) for t in t_list_stage],
            device=device, dtype=dtype
        )
        S = t_list_stage.numel()

        t_floor_stage = t_list_stage[0]
        t_ceil_stage  = t_list_stage[-1]

        if start_t is None:
            start_t = torch.zeros((Bp, 1), device=device, dtype=dtype)
        else:
            if not torch.is_tensor(start_t):
                start_t = torch.tensor(start_t, device=device, dtype=dtype)
            start_t = start_t.to(device=device, dtype=dtype)

            if start_t.ndim == 0:
                start_t = start_t.view(1, 1).expand(Bp, 1)
            elif start_t.ndim == 1:
                start_t = start_t.view(-1, 1)
            if start_t.shape[0] != Bp:
                if start_t.shape[0] == 1:
                    start_t = start_t.expand(Bp, 1)
                else:
                    raise ValueError(f"start_t batch mismatch: got {start_t.shape}, expected ({Bp}, 1)")

        start_idx = torch.searchsorted(t_list_stage, start_t.squeeze(1), right=False) 
        start_idx = start_idx.clamp(0, S - 1)
        
        if one_step:
            t0 = t_list_stage[start_idx].view(Bp, 1)
            y  = t_m(t0, t_ceil_stage)            
            drift = v_m(x_t, y)  
            nd = x_t.ndim
            remaining = (S - 1 - start_idx).to(dtype=dtype).view(Bp, 1, *([1] * (nd - 2)))
            scale = self.delta_t * remaining
            del_x = drift * scale
            x_t = x_t + del_x
            if mask is not None:
                x_t = torch.where(mask, x_1_fixed, x_t)
            return x_t

        max_remain = int((S - 1 - start_idx).max().item())  

        for r in range(max_remain):
            cur_idx = start_idx + r                
            active  = cur_idx < (S - 1)               
            if not torch.any(active):
                break

            act_idx = torch.nonzero(active, as_tuple=False).squeeze(1)

            t_cur = t_list_stage[cur_idx[act_idx]].view(-1, 1) 
            y = t_m(t_cur, t_ceil_stage)
            drift = v_m(x_t[act_idx], y)

            t_norm = (t_cur - t_floor_stage) / (t_ceil_stage - t_floor_stage)
            nd = x_t.ndim
            scheduled_noise = self.beta_t(t_norm).view(-1, 1, *([1] * (nd - 2)))

            z = torch.randn_like(drift) * (self.delta_t ** 0.5)
            if r == max_remain - 1:
                del_x = drift * self.delta_t
            else:
                del_x = drift * self.delta_t + (scheduled_noise ** 0.5) * z

            x_t_act = x_t[act_idx] + del_x
            if mask is not None:
                x_t_act = torch.where(mask[act_idx], x_1_fixed[act_idx], x_t_act)
            x_t = x_t.clone()
            x_t[act_idx] = x_t_act

        return x_t
