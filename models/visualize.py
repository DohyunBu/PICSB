import numpy as np
import matplotlib.pyplot as plt
from models.loss import *
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
import numpy as np
import matplotlib.pyplot as plt
import os

def save_field_snapshot(
    data,
    mask=None,
    batch_idx: int = 0,
    cmap: str = "coolwarm",
    interpolation: str = "nearest",
    vmin: float = -0.35,
    vmax: float = 0.35,
    out_svg: str = "field.svg",
    border_lw: float = 3.0,
    masked_bad_color="white",   
):
    B, C, H, W = data.shape
    assert 0 <= batch_idx < B
    assert C == 1, "C=1 only"

    img = data[batch_idx, 0] 
    
    if out_svg is None:
        os.makedirs(out_dir, exist_ok=True)
        out_svg = os.path.join(out_dir, f"field_b{batch_idx:04d}.svg")
    else:
        out_dir = os.path.dirname(out_svg)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)
            
    if mask is not None:
        m = mask
        if hasattr(m, "detach"):
            m = m.detach().cpu().numpy()
        m = np.asarray(m)

        if m.ndim == 4:     
            m2 = m[batch_idx, 0]
        elif m.ndim == 3:
            m2 = m[batch_idx] if m.shape[0] == B else m[0]
        elif m.ndim == 2:  
            m2 = m
        else:
            raise ValueError(f"Unsupported mask shape: {m.shape}")

        m2 = m2.astype(bool)
        img = np.where(m2, img, np.nan)

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color=masked_bad_color)

    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")
    ax.imshow(
        img,
        origin="upper",
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        aspect="auto",
    )

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(border_lw)
        spine.set_edgecolor("black")

    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(
        out_svg,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="none",
        edgecolor="none",
        dpi=900
    )
    plt.close(fig)