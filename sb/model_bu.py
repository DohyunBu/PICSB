import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import sb.networks
import os
from visualize import *
from loss import *
from IPython.display import clear_output

class sb_muti_model(object):
    def __init__(self,pi_list,timepoints,N_train=1000,B=128,steps=60,eps_train=10,eps_test=10,base_lr=1e-3,pic_min=-1,pic_max=1,patience=1e8,lambda_=1e-3,clamp=False,save=False,mask_list=None, decay=0.8, directory_name ='model_bu/'):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.length=len(pi_list)
        if mask_list is not None and len(mask_list) != self.length:
            raise ValueError("mask_list length must match pi_list length.")
        self.pi=[]
        for i in range(self.length):
            mask_i = None if mask_list is None else mask_list[i]
            self.pi.append(sb.networks.CustomDataset(pi_list[i], mask_i))
        self.timepoints = timepoints
        self.N_train=N_train
        self.B=B
        self.d=pi_list[0].shape[1]
        self.steps=steps
        self.eps_train=eps_train
        self.eps_test=eps_test
        self.base_lr=base_lr
        self.min = pic_min 
        self.max = pic_max
        self.delta_t = (timepoints[-1] - timepoints[0]) / self.steps
        self.patience=patience
        self.lambda_ = lambda_
        self.clamp = clamp
        self.t_list = torch.arange(timepoints[0], timepoints[-1], self.delta_t, device=self.device, dtype=torch.float32)
        self.t_list = torch.cat([self.t_list, torch.tensor([timepoints[-1]], device=self.device, dtype=self.t_list.dtype)])
        self.save=save
        self.decay = decay
        
        
        # self.alpha = nn.Parameter(torch.randn(1, requires_grad=True, device=self.device)/1000)
        self.alpha = nn.Parameter(torch.tensor([1.], device=self.device, requires_grad=True, dtype=torch.float32))
        
        os.makedirs(directory_name, exist_ok=True)
        self.path = directory_name

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
            
        self.v_fore=sb.networks.SimpleUNet(condition_dim=32).to(self.device)
        self.scale_m_fore=sb.networks.scale_model_muti(output_size=32).to(self.device)
        self.optimizer_fore = optim.Adam(list(self.v_fore.parameters())+list(self.scale_m_fore.parameters()), lr=self.base_lr)
        self.alpha_optimizer = optim.SGD([self.alpha], lr=self.base_lr/1000, momentum=0.1)
        
    def _split_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            return batch[0].to(self.device, dtype=torch.float32), batch[1].to(self.device, dtype=torch.float32)
        return batch.to(self.device, dtype=torch.float32), None
          
    def train(self, low_resol_coef, loss_plot=True):

        self.finetuning_loss_history={'loss':[], "loss_fore":[], "PDE_loss_fore":[], "Alpha":[]}
        
        # fore_params = {name: param.clone().detach() for name, param in self.v_fore.named_parameters()}
        # for param in fore_params.values():
        #     param.requires_grad = False  
            
        with tqdm(total=self.N_train) as pbar:
            for n in range(self.N_train):
                dataloader_list=[DataLoader(self.pi[i],shuffle=False,batch_size=self.B,drop_last=True) for i in range(self.length)]
                
                epoch_loss_fore = 0
                epoch_pde_loss_fore = 0
                epoch_pde_loss_fore2 = 0
                
                self.optimizer_fore.zero_grad()
                self.alpha_optimizer.zero_grad()
                for data_tuple in zip(*dataloader_list):
                    num_all=[]
                    for i in range(self.length):
                        batch_i = data_tuple[i][0] if isinstance(data_tuple[i], (list, tuple)) else data_tuple[i]
                        num_all.append(batch_i.shape[0])
                        
                    for stage_index in range(self.length-1):
                        if n < self.N_train / 3 :
                            x_0, mask_0 = self._split_batch(data_tuple[stage_index])
                            x_1, mask_1 = self._split_batch(data_tuple[stage_index+1])
                            x_1_fixed = (x_1 * mask_1).detach()
                            
                            x_0 = torch.where(mask_1.bool(), x_1_fixed, x_0)
                            
                            x_t_fore, t_fore, t_ceil, t_floor = self.Interp_t(x_0, x_1, forward=True)
                            x_t_fore = torch.where(mask_1.bool(), x_1_fixed, x_t_fore)
                            
                            x_fore = self.v_fore(x_t_fore,self.scale_m_fore(t_fore, 1))
                            x_fore = x_fore.view(self.B, 1, 128, 128)
                            
                            target_fore = (x_1 - x_t_fore) / (t_ceil - t_fore).view(-1, 1, 1, 1)
                            
                            diff = (x_fore - target_fore).abs() * (~(mask_1.bool()))
                            loss_fore = diff.sum() / (~(mask_1.bool())).sum()
                            
                            x_1_hat = self.fwd_start_from_t(x_t_fore.view(-1,1,128,128), x_1_fixed, mask_1, self.v_fore, self.scale_m_fore, stage_index, start_t=t_fore.view(-1,1), one_step=False)[-1].cuda()
                            
                            pde_fore = burgers_pde_mse(x_1_hat, coef=torch.abs(self.alpha))
                            pde_fore2 = burgers_pde_mse_observed_pixel(x_1_hat, coef=torch.abs(self.alpha), mask=mask_1)
                            
                            loss = pde_fore
                        else:
                            x_0, mask_0 = self._split_batch(data_tuple[stage_index])
                            x_0 = torch.where(mask_1.bool(), x_1_fixed, x_0)
                            x_1, mask_1 = self._split_batch(data_tuple[stage_index+1])
                            # x_1 = self.fwd_start_from_t(x_0.view(-1,1,128,128), x_1_fixed, mask_1, self.v_fore, self.scale_m_fore, stage_index, one_step=False)[-1].cuda()
                            
                            
                            x_t_fore, t_fore, t_ceil, t_floor = self.Interp_t(x_0, x_1, forward=True)
                            x_t_fore = torch.where(mask_1.bool(), x_1_fixed, x_t_fore)
                            
                            x_fore = self.v_fore(x_t_fore,self.scale_m_fore(t_fore, 1))
                            x_fore = x_fore.view(self.B, 1, 128, 128)
                            
                            target_fore = (x_1 - x_t_fore) / (t_ceil - t_fore).view(-1, 1, 1, 1)
                            
                            diff = (x_fore - target_fore).abs() * (~(mask_1.bool()))
                            loss_fore = diff.sum() / (~(mask_1.bool())).sum()
                            
                            x_1_hat = self.fwd_start_from_t(x_t_fore.view(-1,1,128,128), x_1_fixed, mask_1, self.v_fore, self.scale_m_fore, stage_index, start_t=t_fore.view(-1,1), one_step=False)[-1].cuda()
                            
                            pde_fore = burgers_pde_mse(x_1_hat, coef=torch.abs(self.alpha))
                            pde_fore2 = burgers_pde_mse_observed_pixel(x_1_hat, coef=torch.abs(self.alpha), mask=mask_1)
                            
                            loss = pde_fore
                            
                        epoch_loss_fore = loss_fore
                        epoch_pde_loss_fore = pde_fore
                        epoch_pde_loss_fore2 = pde_fore2
                        loss.backward()
                        self.optimizer_fore.step()
                        self.alpha_optimizer.step()
                        
                if n % 1 == 0:
                    with torch.no_grad():
                        self.finetuning_loss_history['loss_fore'].append(np.log1p(epoch_loss_fore.item()))
                        self.finetuning_loss_history['PDE_loss_fore'].append(np.log1p(epoch_pde_loss_fore.item()))
                        self.finetuning_loss_history['Alpha'].append(torch.abs(self.alpha).item())
                        
                if self.save and n%100==0 and n!=0:
                    models_dict = {
                        'v_fore': self.v_fore.state_dict(),
                        'scale_m_fore': self.scale_m_fore.state_dict(),
                        }
                    torch.cuda.empty_cache()
                    clear_output(wait=True)
                    list3 = self.finetuning_loss_history['Alpha']
                    x = range(len(list3))
                    plt.plot(x, list3, label='Alpha')
                    plt.ylim(0.,1.)
                    plt.legend()
                    plt.show()
                    plt.close()
                    
                pbar.set_description('processed: %d' % (1 + n))
                if True:
                    pbar.set_postfix({'MSE_observed':epoch_loss_fore.detach().cpu().numpy(), 'PDE_loss':epoch_pde_loss_fore.detach().cpu().numpy(), 'PDE_loss2':epoch_pde_loss_fore2.detach().cpu().numpy(), 'Alpha':torch.abs(self.alpha).detach().cpu().numpy()})
                    pbar.update(1)
                
        if self.save :
            models_dict = {
                'v_fore': self.v_fore.state_dict(),
                'scale_m_fore': self.scale_m_fore.state_dict(),
                }
            torch.save(models_dict, self.path+'/finetuning_end.pt')
            torch.cuda.empty_cache()
            
        if loss_plot:
            list1 = self.finetuning_loss_history['loss_fore']
            x = range(len(list1))
            plt.plot(x, list1, label='loss_fore')
            plt.legend()
            plt.show()
            plt.close()
            
            list2 = self.finetuning_loss_history['PDE_loss_fore']
            x = range(len(list2))
            plt.plot(x, list2, label='PDE_loss_fore')
            plt.legend()
            plt.show()
            plt.close()
            
            list3 = self.finetuning_loss_history['Alpha']
            x = range(len(list3))
            plt.plot(x, list3, label='Alpha')
            plt.legend()
            plt.show()
            plt.close()
            
    def finetuning_load(self,model_path='finetuning_end.pt'):
        models_dict=torch.load(model_path)
        self.v_fore=copy.deepcopy(self.v_fore).train()
        self.v_fore.load_state_dict(models_dict['v_fore'])
        
        self.scale_m_fore=copy.deepcopy(self.scale_m_fore).train()
        self.scale_m_fore.load_state_dict(models_dict['scale_m_fore'])
    
    def eval_fore(self, x_0, x_1_fixed, mask, eps_test=None):
        self.v_fore.eval()
        self.scale_m_fore.eval()
        with torch.no_grad():
            x_f=self.fwd_inference(x_0, x_1_fixed, mask, self.v_fore, self.scale_m_fore, eps_test, stage_index=0)
            
        return x_f
        
    def Interp_t(self, x_0, x_1, forward=True):
        B=x_0.shape[0]
        
        stage = np.random.choice(self.length-1)
        t_list = self.t_lists_stage[stage]    
        t_list = torch.tensor([float(t) for t in t_list], device=x_0.device, dtype=x_0.dtype)
        t_floor = t_list[0]
        t_ceil  = t_list[-1]
        if forward:
            t_list = t_list[:-1]
        else:
            t_list = t_list[1:]
        t = t_list[torch.randint(len(t_list), (B,))].reshape(-1, 1)
        Z = torch.randn(B, 1, 128, 128, device=x_0.device, dtype=x_0.dtype)
        

        alpha = (t_ceil - t) / (t_ceil - t_floor)
        beta = (t - t_floor) / (t_ceil - t_floor)
        alpha = alpha.view(B, 1, 1, 1).to(device=x_0.device, dtype=x_0.dtype)
        beta  = beta.view(B, 1, 1, 1).to(device=x_0.device, dtype=x_0.dtype)
        x_t = (alpha * x_0 + beta * x_1 + torch.sqrt(self.eps_train * alpha * beta) * Z)
        
        return x_t, t, t_ceil, t_floor

    def fwd_start_from_t(self, x_0, x_1_fixed, mask, v_m, s_m, stage_index, start_t=None, one_step=False):
        B = x_0.shape[0]
        
        x = []
        x_t = x_0
        x.append(x_t)

        t_list_stage = self.t_lists_stage[stage_index]
        t_ceil_stage = self.t_lists_stage[stage_index][-1]

        start_idx = 0
        if start_t is not None:
            if torch.is_tensor(start_t):
                start_t_val = float(start_t.detach().reshape(-1)[0].cpu().item())
            else:
                start_t_val = float(start_t)

            for i, t_new in enumerate(t_list_stage):
                t_new_val = float(t_new.detach().cpu().item()) if torch.is_tensor(t_new) else float(t_new)
                if t_new_val >= start_t_val - 1e-12:
                    start_idx = i
                    break
        if one_step:
            run_times = t_list_stage[start_idx:start_idx+1]
        else:
            run_times = t_list_stage[start_idx:]
        run_times = [t for t in run_times if t < 1.0]
        
        for k, t_new in enumerate(run_times):
            
            t = t_new.expand(B, 1)
            x_t = x[-1].cuda() 

            y = s_m(t, t_ceil_stage)
            drift = v_m(x_t, y)
            
            if one_step:
                del_x_t = drift * (self.delta_t*(len(t_list_stage)-start_idx-1))
                x_t = x_t + del_x_t.view(B, 1, 128, 128)
                x_t = torch.where(mask.bool(), x_1_fixed, x_t)
                x.append(x_t)
                
                return x
            else:
                if t_new < t_list_stage[-2]:
                    z = torch.randn(B, self.d, device=x_0.device, dtype=x_0.dtype) * (self.delta_t**0.5)
                    del_x_t = drift * self.delta_t + np.sqrt(self.eps_train) * z
                    
                else:
                    del_x_t = drift * self.delta_t
            x_t = x_t + del_x_t.view(B, 1, 128, 128)
            x_t = torch.where(mask.bool(), x_1_fixed, x_t)
            x.append(x_t)
            
        return x

    def fwd_inference(self, x_0, x_1_fixed, mask, v_m, s_m, eps_test=None, stage_index=0):
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
            y = s_m(t, t_ceils_stage)
            drift = v_m(x_t, y)
            if t_new < t_list_stage[-2]:
                z = torch.randn(B, self.d, device=x_0.device, dtype=x_0.dtype) * (self.delta_t**0.5)
                del_x_t = drift * self.delta_t + np.sqrt(eps_test) * z
                
            else:
                del_x_t = drift * self.delta_t
            x_t = x_t.view(B, 1, 128, 128) + del_x_t.view(B, 1, 128, 128)
            x_t = torch.where(mask, x_1_fixed, x_t)
            x.append((x_t.cuda()).detach().cpu())
            
        return x
       
    def apply_ema_to_model(self,model, ema_params):
        for name, param in model.named_parameters():
            ema_params[name].mul_(self.decay).add_(param, alpha=1 - self.decay)
            
            param.data.copy_(ema_params[name])
