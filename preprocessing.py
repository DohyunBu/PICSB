import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import copy

def time_windows(U_tx: np.ndarray, win_T: int, stride: int, start_t: int = 0, end_t: int = None):
    Nt, Nx = U_tx.shape
    end_t = min(end_t, Nt)

    if win_T <= 0 or stride <= 0:
        raise ValueError("win_T and stride must be positive.")
    if start_t < 0 or start_t >= end_t:
        raise ValueError("invalid start_t/end_t.")
    if start_t + win_T > end_t:
        raise ValueError("window longer than available interval.")

    starts = np.arange(start_t, end_t - win_T + 1, stride, dtype=int)
    W = np.stack([U_tx[s:s+win_T, :] for s in starts], axis=0) 
    return W

def get_U_full_tx(data_dict, sample_idx: int = 0, filled=False) -> np.ndarray:
    if filled:
        U = data_dict["U_filled"][sample_idx]
    else:    
        U = data_dict["U_full"][sample_idx] 
    if U.ndim == 3:
        return U[0]
    return U

def make_paired_time_window_dataset(
    dataA, dataB,
    *,
    sample_idx_A: int = 0,
    sample_idx_B: int = 0,
    win_T: int = 128,
    stride: int = 32,
    start_t: int = 0,
    end_t: int = None,
):
    UA = get_U_full_tx(dataA, sample_idx_A)
    UB = get_U_full_tx(dataB, sample_idx_B, filled=True) 

    NtA, NxA = UA.shape
    NtB, NxB = UB.shape
    if NxA != NxB:
        raise ValueError(f"Nx mismatch: {NxA} vs {NxB}")

    Nt_common = min(NtA, NtB)
    WA_np = time_windows(UA[:Nt_common], win_T=win_T, stride=stride, start_t=start_t, end_t=Nt_common)  
    WB_np = time_windows(UB[:Nt_common], win_T=win_T, stride=stride, start_t=start_t, end_t=Nt_common)

    starts = np.arange(start_t, Nt_common - win_T + 1, stride, dtype=int)

    WA = torch.from_numpy(WA_np).unsqueeze(1).float()
    WB = torch.from_numpy(WB_np).unsqueeze(1).float()
    
    return WA, WB, starts

def flatten_batch(X: torch.Tensor) -> torch.Tensor:
    return X.reshape(X.shape[0], -1)

def flatten(WA: torch.Tensor, WB: torch.Tensor):
    A_vec = flatten_batch(WA)
    B_vec = flatten_batch(WB)

    A_scaled = torch.from_numpy(A_vec.cpu().numpy().astype(np.float32))
    B_scaled = torch.from_numpy(B_vec.cpu().numpy().astype(np.float32))
    
    return A_scaled, B_scaled

def fill_missing_with_linear_interp(data):
    data_filled = {k: copy.deepcopy(v) for k, v in data.items()}
    U_obs = data_filled["U_obs"].copy()
    x = data_filled["x"]   
    mask_all = data_filled["mask"]

    if mask_all.ndim == 1:             
        mask_per_sample = np.repeat(mask_all[None, :], U_obs.shape[0], axis=0)
    else:
        mask_per_sample = mask_all

    for sample_idx in range(U_obs.shape[0]):
        mask = mask_per_sample[sample_idx]
        known_x = x[mask]
        Nt = U_obs.shape[2]
        for t_idx in range(Nt):
            known_vals = U_obs[sample_idx, 0, t_idx, mask]
            filled = np.interp(x, known_x, known_vals)
            U_obs[sample_idx, 0, t_idx, :] = filled

    data_filled["U_filled"] = U_obs.copy()
    
    return data_filled