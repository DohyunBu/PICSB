import numpy as np
import matplotlib.pyplot as plt
import torch
from loss import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
print('dfdsaf')
def load_burgers_npz(path: str):
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["nu"] = float(np.array(out["nu"]).reshape(-1)[0])
    return out

def plot_full_and_observed_heatmaps(
    data,
    sample_idx: int = 0,
    vmin=None,
    vmax=None,
    cmap: str = "turbo",
):
    x = data["x"]
    t = data["t"]
    nu = data.get("nu", None)

    U_full_i = data["U_full"][sample_idx]
    U_obs_i  = data["U_obs"][sample_idx]
    U_filled_i  = data["U_filled"][sample_idx]
    
    if U_full_i.ndim == 3:
        U = U_full_i[0]
    else:                 
        U = U_full_i

    if U_obs_i.ndim == 3:  
        Uobs = U_obs_i[0]
    else:
        Uobs = U_obs_i
        
    if U_filled_i.ndim == 3:  
        Ufilled = U_filled_i[0]
    else:
        Ufilled = U_filled_i
        
    vmin = np.nanmin(U)
    vmax = np.nanmax(U)

    extent = [x.min(), x.max(), t.max(), t.min()]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(
        U,
        aspect="auto",
        origin="upper",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    title0 = f"Full grid"
    if nu is not None:
        title0 += f" (ν={float(nu):g})"
    title0 += f", sample={sample_idx}"
    axes[0].set_title(title0)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    Uobs_ma = np.ma.masked_invalid(Uobs)
    im1 = axes[1].imshow(
        Uobs_ma,
        aspect="auto",
        origin="upper",
        extent=extent,  
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    axes[1].set_title("Observed Sensors")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(
        Ufilled,
        aspect="auto",
        origin="upper",
        extent=extent,  
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    axes[2].set_title("Interploated Observed Sensors")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)
    plt.show()

def plot_result(
    dataA, dataB,
    predict,
    cmap="turbo",
    high_resol_coef=None,
    low_resol_coef=None,
    mask_1d=None,
):
    
    after = (
        predict.detach().cpu().numpy()
        if torch.is_tensor(predict)
        else np.asarray(predict)
    )

    gen_after  = after[0]  if after.ndim == 3  else after

    gtA = dataA[0][0]
    gtB = dataB[0][0]

    gtA = np.asarray(gtA)
    gtB = np.asarray(gtB)
    gen_after  = np.asarray(gen_after)
        

    extent = [0, 1, 1, 0]

    all_vals = np.concatenate([gtA.ravel(), gtB.ravel()])
    vmin_gt, vmax_gt = np.nanmin(all_vals), np.nanmax(all_vals)

    err_after  = np.abs(gtB - gen_after)
    
    if low_resol_coef is not None:
        pde_map_predict = burgers_pde_all_pixel(predict.unsqueeze(0).unsqueeze(0) , low_resol_coef)[0][0].detach().cpu().numpy()
        pde_map_gta = burgers_pde_all_pixel(torch.tensor(gtA).unsqueeze(0).unsqueeze(0), high_resol_coef)[0][0].detach().cpu().numpy()
        pde_map_gtb = burgers_pde_all_pixel(torch.tensor(gtB).unsqueeze(0).unsqueeze(0), low_resol_coef)[0][0].detach().cpu().numpy()
    pde_map = np.abs(pde_map_predict - pde_map_gtb)
    
    finite_err = err_after[np.isfinite(err_after)]
    vmin_e = 0.0
    vmax_e = np.percentile(finite_err, 99) if finite_err.size > 0 else 1.0
    
    finite_pde_gta = pde_map_gta[1:-1, 2:-2][np.isfinite(pde_map_gta[1:-1, 2:-2])]
    vmin_gta = 0.0
    vmax_gta = np.percentile(finite_pde_gta, 99) if finite_pde_gta.size > 0 else 1.0 
    
    finite_pde_gtb = pde_map_gtb[1:-1, 2:-2][np.isfinite(pde_map_gtb[1:-1, 2:-2])]
    vmin_gtb = 0.0
    vmax_gtb = np.percentile(finite_pde_gtb, 99) if finite_pde_gtb.size > 0 else 1.0 
    
    finite_pde_predict = pde_map_predict[1:-1, 2:-2][np.isfinite(pde_map_predict[1:-1, 2:-2])]
    vmin_predict = 0.0
    vmax_predict = np.percentile(finite_pde_predict, 99) if finite_pde_predict.size > 0 else 1.0 
    
    if mask_1d is not None:
        mask = np.asarray(mask_1d, dtype=bool).reshape(-1)
        err_after  = err_after.copy()
        err_after[:,  ~mask] = np.nan
        pde_map  = pde_map.copy()
        pde_map[:,  ~mask] = np.nan


    
    cmap_err = plt.cm.get_cmap(cmap).copy()
    cmap_err.set_bad(color="white")

    fig, axes = plt.subplots(2, 4, figsize=(15, 5))

    im_gtA = axes[0, 0].imshow(
        gtA, origin="upper", extent=extent,
        aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
    )
    axes[0, 0].set_title("GT A", fontsize=10)

    im_gtB = axes[0, 1].imshow(
        gtB, origin="upper", extent=extent,
        aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
    )
    axes[0, 1].set_title("GT B", fontsize=10)

    im_generated = axes[0, 2].imshow(
        gen_after, origin="upper", extent=extent,
        aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
    )
    axes[0, 2].set_title("Generated B", fontsize=10)

    im_err_after = axes[0, 3].imshow(
        err_after, origin="upper", extent=extent,
        aspect="equal", cmap=cmap_err, vmin=vmin_e, vmax=vmax_e,
        interpolation="nearest"
    )
    axes[0, 3].set_title("Masked error", fontsize=10)
    axes[0, 3].set_facecolor("white")
        
    im_pde_gta = axes[1, 0].imshow(
        pde_map_gta, origin="upper", extent=extent,
        aspect="equal", cmap=cmap_err, vmin=vmin_gta, vmax=vmax_gta,
        interpolation="nearest"
    )
    axes[1, 0].set_title("GT A PDE error", fontsize=10)
    axes[1, 0].set_facecolor("white")
    
    im_pde_gtb = axes[1, 1].imshow(
        pde_map_gtb, origin="upper", extent=extent,
        aspect="equal", cmap=cmap_err, vmin=vmin_gtb, vmax=vmax_gtb,
        interpolation="nearest"
    )
    axes[1, 1].set_title("GT B PDE error", fontsize=10)
    axes[1, 1].set_facecolor("white")
    
    im_pde_predict = axes[1, 2].imshow(
        pde_map_predict, origin="upper", extent=extent,
        aspect="equal", cmap=cmap_err, vmin=vmin_predict, vmax=vmax_predict,
        interpolation="nearest"
    )
    axes[1, 2].set_title("Generated B PDE error", fontsize=10)
    axes[1, 2].set_facecolor("white")
    
    im_pde_err = axes[1, 3].imshow(
        pde_map, origin="upper", extent=extent,
        aspect="equal", cmap=cmap_err, vmin=vmin_predict, vmax=vmax_predict,
        interpolation="nearest"
    )
    axes[1, 3].set_title("PDE error GT vs Generated", fontsize=10)
    axes[1, 3].set_facecolor("white")
    # for ax in axes:
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("t")
    div0 = make_axes_locatable(axes[0, 0])
    cax_gt = div0.append_axes("right", size="6%", pad=0.08)
    cb_gt = fig.colorbar(im_gtB, cax=cax_gt)
    cb_gt.set_label("Field value", fontsize=10)
    
    div1 = make_axes_locatable(axes[0, 1])
    cax_gt = div1.append_axes("right", size="6%", pad=0.08)
    cb_gt = fig.colorbar(im_gtB, cax=cax_gt)
    cb_gt.set_label("Field value", fontsize=10)
    
    div2 = make_axes_locatable(axes[0, 2])
    cax_gt = div2.append_axes("right", size="6%", pad=0.08)
    cb_gt = fig.colorbar(im_gtB, cax=cax_gt)
    cb_gt.set_label("Field value", fontsize=10)

    div3 = make_axes_locatable(axes[0, 3])
    cax_err = div3.append_axes("right", size="6%", pad=0.08)
    cb_err = fig.colorbar(im_err_after, cax=cax_err)
    cb_err.set_label("Masked error", fontsize=10)

    div4 = make_axes_locatable(axes[1, 0])
    cax_pde = div4.append_axes("right", size="6%", pad=0.08)
    cb_pde = fig.colorbar(im_pde_gta, cax=cax_pde)
    cb_pde.set_label("PDE residual", fontsize=10)
    
    div5 = make_axes_locatable(axes[1, 1])
    cax_pde = div5.append_axes("right", size="6%", pad=0.08)
    cb_pde = fig.colorbar(im_pde_gtb, cax=cax_pde)
    cb_pde.set_label("PDE residual", fontsize=10)
    
    div6 = make_axes_locatable(axes[1, 2])
    cax_pde = div6.append_axes("right", size="6%", pad=0.08)
    cb_pde = fig.colorbar(im_pde_predict, cax=cax_pde)
    cb_pde.set_label("PDE residual", fontsize=10)
    
    div7 = make_axes_locatable(axes[1, 3])
    cax_pde = div7.append_axes("right", size="6%", pad=0.08)
    cb_pde = fig.colorbar(im_pde_err, cax=cax_pde)
    cb_pde.set_label("PDE error", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()



    
def plot_train_process(
    dataA,
    dataB,
    stitched,
    sample_idxs=None,
    gen_label="Generated (stitched)",
    cmap="turbo",
    nT=128
):
    dataA = dataA.view(nT, 128)
    dataB = dataB.view(nT, 128)
    dataA_np = dataA.detach().cpu().numpy() if torch.is_tensor(dataA) else np.asarray(dataA)
    if dataA_np.ndim == 2:
        dataA_np = dataA_np[None, ...]  

    dataB_np = dataB.detach().cpu().numpy() if torch.is_tensor(dataB) else np.asarray(dataB)
    if dataB_np.ndim == 2:
        dataB_np = dataB_np[None, ...]  
    stitched_np = stitched.detach().cpu().numpy() if torch.is_tensor(stitched) else np.asarray(stitched)
    if stitched_np.ndim == 2:
        stitched_np = stitched_np[None, ...]  


    fig, axes = plt.subplots(1, 6, figsize=(18, 3 * 1), constrained_layout=False)
    if 1 == 1:
        axes = np.expand_dims(axes, axis=0)

    all_vals = []

    for row, idx in enumerate(sample_idxs):
        gen1 = dataA_np[row] if dataA_np.shape[0] == 1 else dataA_np[idx]
        gen2 = dataB_np[row] if dataB_np.shape[0] == 1 else dataB_np[idx]

        all_vals.append(gen1.ravel())
        all_vals.append(gen2.ravel())

    all_vals = np.concatenate(all_vals)
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    
    for row, idx in enumerate(sample_idxs):
        gen3 = stitched_np[row] if stitched_np.shape[0] == 1 else stitched_np[idx]

        
        im0 = axes[row, 0].imshow(
            gen1, origin="upper",  aspect="auto", vmin=vmin, vmax=vmax,
            cmap=cmap, interpolation="nearest"
        )
        axes[row, 0].set_title(f"GT A (idx={idx})")
        axes[row, 0].set_xlabel("x")
        axes[row, 0].set_ylabel("t")

        im1 = axes[row, 1].imshow(
            gen2, origin="upper",  aspect="auto", vmin=vmin, vmax=vmax,
            cmap=cmap,  interpolation="nearest"
        )
        axes[row, 1].set_title(f"GT B (idx={idx})")
        axes[row, 1].set_xlabel("x")
        axes[row, 1].set_ylabel("t")

        im2 = axes[row, 2].imshow(
            gen3, origin="upper", aspect="auto", vmin=vmin, vmax=vmax,
            cmap=cmap,  interpolation="nearest"
        )
        axes[row, 2].set_title(gen_label)
        axes[row, 2].set_xlabel("x")
        axes[row, 2].set_ylabel("t")
        
        im3 = axes[row, 3].imshow(
            np.abs(gen3-gen1), origin="upper", aspect="auto", vmin=0, vmax=max(vmax,-vmin),
            cmap=cmap,  interpolation="nearest"
        )
        axes[row, 3].set_title("Predict - GT A")
        axes[row, 3].set_xlabel("x")
        axes[row, 3].set_ylabel("t")
        
        im4 = axes[row, 4].imshow(
            np.abs(gen3-gen2), origin="upper", aspect="auto", vmin=0, vmax=max(vmax,-vmin),
            cmap=cmap,  interpolation="nearest"
        )
        axes[row, 4].set_title("Predict - GT B")
        axes[row, 4].set_xlabel("x")
        axes[row, 4].set_ylabel("t")

        im5 = axes[row, 5].imshow(
            np.abs(gen1-gen2), origin="upper", aspect="auto", vmin=0, vmax=max(vmax,-vmin),
            cmap=cmap,  interpolation="nearest"
        )
        axes[row, 5].set_title("GT A - GT B")
        axes[row, 5].set_xlabel("x")
        axes[row, 5].set_ylabel("t")
        
    fig.colorbar(im1)
    fig.colorbar(im5)

    plt.show()
    
def stitch_windows(gen_vectors, starts, win_T=128, Nx=128, stride=128):
    nW = len(starts)
    Nt = starts[-1] + win_T
    full = torch.zeros(Nt, Nx)
    for i in range(nW):
        block = gen_vectors[i].reshape(win_T, Nx)
        start = starts[i]
        full[start:start+win_T] = block
    return full