import numpy as np
import matplotlib.pyplot as plt
import torch
from sb import loss
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

import numpy as np
import matplotlib.pyplot as plt

def dataset_visualization(
    data,
    mask,
    batch_idx: int = 0,
    cmap: str = "turbo",
    interpolation: str = "nearest",
):

    B, C, H, W = data.shape
    assert 0 <= batch_idx < B, f"batch_idx out of range: {batch_idx} (B={B})"

    mask_bool = mask.astype(bool)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="white")
    if C == 1:
        full = data[batch_idx, 0]    
        m = mask_bool[batch_idx, 0]        

        vmin = np.nanmin(full)
        vmax = np.nanmax(full)
        masked_nan = np.where(m, full, np.nan)
        masked  = np.ma.masked_invalid(masked_nan)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        im0 = axes[0].imshow(
            full, aspect="auto", origin="upper",
            vmin=vmin, vmax=vmax, cmap=cm, interpolation=interpolation
        )
        axes[0].set_title(f"Full (b={batch_idx})")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("T")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(
            masked, aspect="auto", origin="upper",
            vmin=vmin, vmax=vmax, cmap=cm, interpolation=interpolation
        )
        axes[1].set_title("Masked")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("T")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.show()
        return

    T = C
    t_idxs = [0, T // 2, T - 1]

    full_sample = data[batch_idx] 
    vmin = np.nanmin(full_sample)
    vmax = np.nanmax(full_sample)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

    for j, ti in enumerate(t_idxs):
        full_img = data[batch_idx, ti] 
        m = mask_bool[batch_idx, ti]          

        masked_nan = np.where(m, full_img, np.nan)
        masked  = np.ma.masked_invalid(masked_nan)

        im_top = axes[0, j].imshow(
            full_img, origin="upper",
            vmin=vmin, vmax=vmax, cmap=cm, interpolation=interpolation
        )
        axes[0, j].set_title(f"Full t={ti} (b={batch_idx})")
        axes[0, j].set_xlabel("Y")
        axes[0, j].set_ylabel("X")

        im_bot = axes[1, j].imshow(
            masked, origin="upper",
            vmin=vmin, vmax=vmax, cmap=cm, interpolation=interpolation
        )
        axes[1, j].set_title(f"Masked t={ti}")
        axes[1, j].set_xlabel("Y")
        axes[1, j].set_ylabel("X")

        fig.colorbar(im_top, ax=axes[0, j], fraction=0.046, pad=0.04)
        fig.colorbar(im_bot, ax=axes[1, j], fraction=0.046, pad=0.04)

    plt.show()