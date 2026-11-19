import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

def dataset_interpolation(A: np.ndarray, mask: np.ndarray, *, interpol_target :str = 'kolmo', noise_levels=None, seed: int=0) -> np.ndarray:

    B, T, X, Y = A.shape
    m = mask.astype(bool)
    xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    A_filled = A.copy()
    
    rng = np.random.default_rng(seed)
    
    for b in tqdm(range(B)):
        for t in range(T):
            known = m[b, t] 
            if known.all():
                continue 

            vals = A[b, t]
            if (interpol_target=='darcy') or (interpol_target=='RD') or (interpol_target=='reaction_diffusion'):
                pts_known = np.stack([xx[known], yy[known]], axis=1)
                pts_all   = np.stack([xx.ravel(), yy.ravel()], axis=1)
                v_known   = vals[known]
                if noise_levels is not None:
                    sigma = np.std(v_known)
                    if sigma > 0:
                        v_known = v_known + rng.normal(0.0, noise_levels * sigma, size=v_known.shape)
                filled = griddata(pts_known, v_known, pts_all, method="nearest").reshape(X, Y)
                vals_out = vals.copy()
                vals_out[known] = v_known
                vals_out[~known] = filled[~known]
                A_filled[b, t] = vals_out
                continue

            pts_known = np.stack([xx[known], yy[known]], axis=1)  
            v_known   = vals[known]                            
            pts_all   = np.stack([xx.ravel(), yy.ravel()], axis=1)
            if noise_levels is not None:
                sigma = np.std(v_known)
                if sigma > 0:
                    v_known = v_known + rng.normal(0.0, noise_levels * sigma, size=v_known.shape)
            filled_cubic = griddata(pts_known, v_known, pts_all, method="cubic").reshape(X, Y)

            vals_out = vals.copy()
            vals_out[known] = v_known 
            vals_out[~known] = filled_cubic[~known]
            A_filled[b, t] = vals_out
    return A_filled
