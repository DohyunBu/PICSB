import numpy as np
import torch

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

