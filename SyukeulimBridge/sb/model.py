import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import sb.networks
import os
from visualize import *
from loss import *
from IPython.display import clear_output
import torch
import torch.nn.functional as F

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

def _prepare_unet_input(low_data, high_data, mask_low, mask_high, device):
    low_data  = _to_device_float(low_data, device)
    high_data = _to_device_float(high_data, device)
    mask_low = _to_device_bool(mask_low, device)
    mask_high = _to_device_bool(mask_high, device)
    B, C, H, W = low_data.shape

    if C == 1:
        if mask_high is not None:
            x1_fixed = (high_data * mask_high).detach()
            low_data = torch.where(mask_high, x1_fixed, low_data)
            high_data = torch.where(mask_high, x1_fixed, high_data)

        return low_data, high_data, mask_high, "1d"

    else:
        T = C
        X, Y = H, W

        low_  = low_data.view(B * T, 1, X, Y)
        high_ = high_data.view(B * T, 1, X, Y)
        mask_ = mask_high.view(B * T, 1, X, Y) if mask_high is not None else None

        if mask_ is not None:
            x1_fixed = (high_ * mask_).detach()
            low_ = torch.where(mask_, x1_fixed, low_)
            high_ = torch.where(mask_, x1_fixed, high_)

        return low_, high_, mask_, "2d"
def _infer_prepare(self, x0, x1_fixed, mask_low, mask_high):
    x0_fix, x1_fix, mask_fix, mode = _prepare_unet_input(
        x0, x1_fixed, mask_low, mask_high, self.device
    )

    B, C, H, W = x0.shape
    meta = {"mode": mode, "B": B, "C": C, "H": H, "W": W}

    return x0_fix, x1_fix, mask_fix, meta


def _infer_restore(self, x_path, meta):
    mode = meta["mode"]
    B, C, H, W = meta["B"], meta["C"], meta["H"], meta["W"]

    if mode == "1d":
        return x_path

    T, X, Y = C, H, W
    restored = []
    for xt in x_path:
        xt = xt.view(B, T, X, Y)
        restored.append(xt)
    return restored

class SB_model_ver1(object):
    def __init__(self, dataloader, timepoints, time_dim=32, steps=60, eps_train=10, eps_test=10, base_lr=1e-3,save=False, decay=0.8):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dataloader = dataloader
        self.timepoints = timepoints
        self.steps=steps
        self.eps_train=eps_train
        self.eps_test=eps_test
        self.base_lr=base_lr
        self.delta_t = (timepoints[-1] - timepoints[0]) / self.steps
        
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
            
        self.save=save
        self.decay = decay
        
        # 1d의 경우 1채널, 2d의 경우 B*T로 1채널로 만듬
        in_channels = 1
        self.v_fore=sb.networks.SimpleUNet(input_dim=in_channels, condition_dim=time_dim).to(self.device)
        self.time_embed=sb.networks.time_embedding(output_dim=time_dim).to(self.device)
        self.optimizer_fore = optim.Adam(list(self.v_fore.parameters())+list(self.time_embed.parameters()), lr=self.base_lr)
        log_alpha_init = torch.empty(1).uniform_(np.log(1e-4), np.log(1e-2))
        self.alpha = nn.Parameter(log_alpha_init.to(self.device))
        self.alpha_optimizer = optim.SGD([self.alpha], lr=self.base_lr, momentum=0.5)
        
        self.path = "model/"
        os.makedirs(self.path, exist_ok=True)
        
    def train(self, train_epochs=10000, loss_plot=True):
        self.finetuning_loss_history={'loss':[], "loss_fore":[], "PDE_loss_fore":[], "Alpha":[]}

        self.v_fore.train()
        self.time_embed.train() 
        loader_it = iter(self.dataloader)

        pbar = tqdm(range(train_epochs), desc="Train (1 batch per epoch)")
        for epoch in pbar:
            try:
                batch = next(loader_it)
            except StopIteration:
                loader_it = iter(self.dataloader)
                batch = next(loader_it)
            self.optimizer_fore.zero_grad(set_to_none=True)
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha = F.softplus(self.alpha)
            x_0, x_1, mask_low, mask_high = batch
            x0_fixed, x1_fixed, mask, mode = _prepare_unet_input(x_0, x_1, mask_low, mask_high, self.device)
            x_1_hat_fixed = self.fwd_start_from_t(x_0=x0_fixed, x_1_fixed=x1_fixed, mask=mask, v_m=self.v_fore, t_m=self.time_embed, one_step=False)[-1]
            x_t_fore, t_fore, t_ceil, t_floor = self.Interp_t(x_0=x0_fixed, x_1=x_1_hat_fixed, forward=True, stage_index=0)
            
            x_fore = self.v_fore(x_t_fore, self.time_embed(t_fore, 1))
            
            target_fore = (x1_fixed - x_t_fore) / (t_ceil - t_fore).view(-1, 1, 1, 1)
            
            if mask is None:
                loss_fore = (x_fore - target_fore).abs().mean()
            else:
                unobs = (~mask.bool())
                diff = (x_fore - target_fore).abs()
                denom = unobs.sum().clamp_min(1)
                loss_fore = (diff * unobs).sum() / denom
            
            x_1_hat = self.fwd_start_from_t(x_0=x_t_fore, x_1_fixed=x1_fixed, mask=mask, v_m=self.v_fore, t_m=self.time_embed, start_t=t_fore, one_step=False)[-1]
            pde_fore = burgers_pde_mse(x_1_hat, coef=alpha)
            
            loss = 1 * loss_fore + 1 * pde_fore
            epoch_loss_fore = loss_fore
            epoch_pde_loss_fore = pde_fore

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v_fore.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(self.time_embed.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_([self.alpha], max_norm=0.1)
            self.optimizer_fore.step()
            self.alpha_optimizer.step()  
            
    def finetuning_load(self,model_path='finetuning_end.pt'):
        models_dict=torch.load(model_path)
        self.v_fore=copy.deepcopy(self.v_fore).train()
        self.v_fore.load_state_dict(models_dict['v_fore'])
        
        self.time_embed=copy.deepcopy(self.time_embed).train()
        self.time_embed.load_state_dict(models_dict['time_embed'])
    
    def eval_fore(self, x_0, x_1_fixed, mask_low=None, mask_high=None, eps_test=None, stage_index=0):
        self.v_fore.eval()
        self.time_embed.eval()

        with torch.no_grad():
            x0_fix, x1_fix, mask_fix, meta = self._infer_prepare(x_0, x_1_fixed, mask_low, mask_high)

            x_path_fix = self.fwd_inference(x_0=x0_fix, x_1_fixed=x1_fix, mask=mask_fix, v_m=self.v_fore, s_m=self.time_embed, eps_test=eps_test, stage_index=stage_index)
            x_path = self._infer_restore(x_path_fix, meta)

            return x_path

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

        alpha = alpha.view(Bp, 1, 1, 1)
        beta  = beta.view(Bp, 1, 1, 1)

        x_t = alpha * x_0 + beta * x_1 + torch.sqrt(torch.tensor(self.eps_train, device=device, dtype=dtype) * alpha * beta) * Z

        return x_t, t, t_ceil, t_floor
    
    def fwd_start_from_t(self, x_0, x_1_fixed, mask, v_m, t_m, stage_index: int=0, start_t=None, one_step=False):
        device = x_0.device
        dtype  = x_0.dtype
        Bp = x_0.shape[0]

        x_t = x_0
        x_path = [x_t]

        t_list_stage = self.t_lists_stage[stage_index]   
        t_ceil_stage = self.t_lists_stage[stage_index][-1]  

        start_idx = 0
        if start_t is not None:
            start_t_val = float(start_t.detach().reshape(-1)[0].cpu().item()) if torch.is_tensor(start_t) else float(start_t)
            for i, t_new in enumerate(t_list_stage):
                t_new_val = float(t_new.detach().cpu().item()) if torch.is_tensor(t_new) else float(t_new)
                if t_new_val >= start_t_val - 1e-12:
                    start_idx = i
                    break

        if one_step:
            run_times = t_list_stage[start_idx:start_idx + 1]
        else:
            run_times = t_list_stage[start_idx:]
        run_times = [t for t in run_times if float(t.detach().cpu().item()) < 1.0]

        for t_new in run_times:
            t = t_new.to(device=device, dtype=dtype).expand(Bp, 1)
            y = t_m(t, t_ceil_stage)  
            drift = v_m(x_t, y)
            if one_step:
                scale = self.delta_t * (len(t_list_stage) - start_idx - 1)
                del_x = drift * scale
                x_t = x_t + del_x
                x_t = torch.where(mask, x_1_fixed, x_t)
                x_path.append(x_t)
                return x_path
            else:
                if t_new < t_list_stage[-1]:
                    z = torch.randn_like(drift) * (self.delta_t ** 0.5)
                    del_x = drift * self.delta_t + (self.eps_train ** 0.5) * z
                    
                else:
                    del_x = drift * self.delta_t
            x_t = x_t + del_x
            x_t = torch.where(mask, x_1_fixed, x_t)
            x_path.append(x_t)

        return x_path

    def fwd_inference(self, x_0, x_1_fixed, mask, v_m, t_m, eps_test=None, stage_index=0):
        B=x_0.shape[0]
        x_0 = x_0.to(self.device, dtype=torch.float32)
        x_1_fixed = x_1_fixed.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device, dtype=torch.float32)
        x_0 = torch.where(mask.bool(), x_1_fixed, x_0.reshape(self.B, 1, 128, 128))
        x_0 = x_0.reshape(self.B, 1, -1)
        mask = mask.to(self.device)
        mask = mask.bool()
        
        x=[]
        x_t=x_0
        x.append(x_t.detach().cpu())
        eps_test = self.eps_test if eps_test is None else eps_test
        
        t_list_stage=self.t_lists_stage[stage_index]
        t_ceils_stage=self.t_lists_stage[stage_index][-1]
        for t_new in t_list_stage[:-1]:
            t = t_new.expand(B, 1)
            x_t=x[-1].cuda()
            y = t_m(t, t_ceils_stage)
            drift = v_m(x_t, y)
            if t_new < t_list_stage[-2]:
                z = torch.randn(B, drift.shape[1], device=x_0.device, dtype=x_0.dtype) * (self.delta_t**0.5)
                del_x_t = drift * self.delta_t + np.sqrt(eps_test) * z
                
            else:
                del_x_t = drift * self.delta_t
            x_t = x_t.view(B, 1, 128, 128) + del_x_t.view(B, 1, 128, 128)
            x_t = torch.where(mask, x_1_fixed, x_t)
            x.append((x_t.cuda()).detach().cpu())
            
        return x
    
    def fwd_inference(self, x0_fix, x1_fix, mask_fix, v_m, t_m, eps_test=None, stage_index=0):
        device = self.device
        dtype = torch.float32

        x_t = x0_fix.to(device=device, dtype=dtype)
        x1 = x1_fix.to(device=device, dtype=dtype)
        mask = mask_fix.to(device=device, dtype=torch.bool) if mask_fix is not None else None

        if mask is not None:
            x_t = torch.where(mask, x1, x_t)

        eps = self.eps_test if eps_test is None else eps_test

        t_list_stage = self.t_lists_stage[stage_index]
        Bp = x_t.shape[0]

        x_path = [x_t.detach().cpu()]

        for t_new in t_list_stage[:-1]:
            t = t_new.to(device=device, dtype=dtype).expand(Bp, 1)
            cond = t_m(t, 1)
            drift = v_m(x_t, cond) 

            if t_new < t_list_stage[-2]:
                z = torch.randn_like(drift) * (self.delta_t ** 0.5)
                del_x = drift * self.delta_t + (eps ** 0.5) * z
            else:
                del_x = drift * self.delta_t
            x_t = x_t + del_x
            if mask is not None:
                x_t = torch.where(mask, x1, x_t)
            x_path.append(x_t.detach().cpu())

        return x_path
    
    def apply_ema_to_model(self,model, ema_params):
        for name, param in model.named_parameters():
            ema_params[name].mul_(self.decay).add_(param, alpha=1 - self.decay)
            
            param.data.copy_(ema_params[name])
