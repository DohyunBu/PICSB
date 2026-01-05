import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import copy
from scipy.interpolate import CubicSpline, interp1d

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

def fill_missing_with_cubic_interp(data):
    data_filled = {k: copy.deepcopy(v) for k, v in data.items()}
    U_obs = np.asarray(data_filled["U_obs"]).copy() 
    x = np.asarray(data_filled["x"]).ravel()         
    m = np.asarray(data_filled["mask"]).astype(bool)  

    B, C, Nt, Nx = U_obs.shape

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]

    for sample_idx in range(B):
        for t_idx in range(Nt):
            mask_x = m[t_idx].ravel()           
            known_x = x[mask_x].ravel()
            if known_x.size < 2:
                continue

            known_vals = U_obs[sample_idx, 0, t_idx, mask_x].ravel()
            if known_vals.size != known_x.size:
                raise ValueError("known_x and known_vals size mismatch")

            ksort = np.argsort(known_x)         
            known_x_sorted = known_x[ksort]
            known_vals_sorted = known_vals[ksort]

            ux, inv = np.unique(known_x_sorted, return_inverse=True)
            if ux.size != known_x_sorted.size:
                tmp = np.zeros_like(ux, dtype=float)
                cnt = np.zeros_like(ux, dtype=float)
                for j, g in enumerate(inv):
                    tmp[g] += known_vals_sorted[j]
                    cnt[g] += 1
                known_x_use = ux
                known_vals_use = tmp / np.maximum(cnt, 1.0)
            else:
                known_x_use = known_x_sorted
                known_vals_use = known_vals_sorted

            if known_x_use.size >= 4:
                cs = CubicSpline(known_x_use, known_vals_use, bc_type="not-a-knot", extrapolate=True)
                filled_sorted = cs(x_sorted)
            else:
                f = interp1d(known_x_use, known_vals_use, kind="linear",
                             fill_value="extrapolate", assume_sorted=True)
                filled_sorted = f(x_sorted)

            filled = np.empty_like(filled_sorted)
            filled[sort_idx] = filled_sorted
            U_obs[sample_idx, 0, t_idx, :] = filled

    data_filled["U_filled"] = U_obs.copy()
    return data_filled
