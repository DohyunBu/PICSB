import numpy as np
import torch
from scipy.interpolate import griddata
from tqdm import tqdm
def time_windows(data: np.ndarray, win_T: int, stride: int, start_t: int = 0, end_t: int = None):
    U = np.asarray(data)
    B, C, Nt, Nx = U.shape
    if end_t is None:
        end_t = Nt
    end_t = min(int(end_t), Nt)
    starts = np.arange(start_t, end_t - win_T + 1, stride, dtype=int)
    W = np.stack([U[:, :, s:s + win_T, :] for s in starts], axis=0)  
    if B == 1:
        W = W[:, 0]
    else:
        W = W.reshape(-1, C, win_T, Nx)

    return W

def make_paired_time_window_dataset(
    low_fidelity, high_fidelity,
    *,
    win_T: int = 128,
    stride: int = 32,
):
    Nt_common = min(low_fidelity.shape[2], high_fidelity.shape[2])

    low_fidelity  = time_windows(low_fidelity,  win_T=win_T, stride=stride, start_t=0, end_t=Nt_common)
    high_fidelity = time_windows(high_fidelity, win_T=win_T, stride=stride, start_t=0, end_t=Nt_common)

    low_fidelity  = torch.from_numpy(low_fidelity).float()
    high_fidelity = torch.from_numpy(high_fidelity).float()

    return low_fidelity, high_fidelity

def bicubic_fill_xy(A: np.ndarray, mask: np.ndarray, *, min_points: int = 16) -> np.ndarray:

    B, T, X, Y = A.shape
    m = mask.astype(bool)
    xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    A_filled = A.copy()
    for b in tqdm(range(B)):
        for t in range(T):
            known = m[b, t] 
            if known.all():
                continue 

            vals = A[b, t]
            n_known = int(known.sum())
            
            if n_known < min_points:
                pts_known = np.stack([xx[known], yy[known]], axis=1)
                pts_all   = np.stack([xx.ravel(), yy.ravel()], axis=1)
                v_known   = vals[known]

                filled = griddata(pts_known, v_known, pts_all, method="nearest").reshape(X, Y)
                vals_out = vals.copy()
                vals_out[~known] = filled[~known]
                A_filled[b, t] = vals_out
                continue

            pts_known = np.stack([xx[known], yy[known]], axis=1)  
            v_known   = vals[known]                            
            pts_all   = np.stack([xx.ravel(), yy.ravel()], axis=1)

            filled_cubic = griddata(pts_known, v_known, pts_all, method="cubic").reshape(X, Y)

            need = ~known
            holes = need & np.isnan(filled_cubic)
            if holes.any():
                filled_nearest = griddata(pts_known, v_known, pts_all, method="nearest").reshape(X, Y)
                filled_cubic[holes] = filled_nearest[holes]

            vals_out = vals.copy()
            vals_out[need] = filled_cubic[need]
            A_filled[b, t] = vals_out
    return A_filled

# low_fideldity_data = bicubic_fill_xy(high_fideldity_data, mask_high_1d)

