import numpy as np
import matplotlib.pyplot as plt
import torch
from loss import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

def load_burgers_npz(path: str):
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["nu"] = float(np.array(out["nu"]).reshape(-1)[0])
    return out

def plot_full_and_observed_heatmaps(
    data,
    sample_idx: int = 0,
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
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.show()

def plot_result(
    dataA, dataB, dataB_predict_initial,
    predict,
    cmap="turbo",
    high_resol_coef=None,
    low_resol_coef=None,
    estimated_coef=None,
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
    dataB_predict_initial = dataB_predict_initial[0][0]
    
    gtA = np.asarray(gtA)
    gtB = np.asarray(gtB)
    dataB_predict_initial = np.asarray(dataB_predict_initial)
    gen_after  = np.asarray(gen_after)
        

    extent = [0, 1, 1, 0]

    all_vals = np.concatenate([gtA.ravel(), gtB.ravel()])
    vmin_gt, vmax_gt = np.nanmin(all_vals), np.nanmax(all_vals)

    err_after  = np.abs(gtB - gen_after)
    
    pde_map_gta = burgers_pde_all_pixel(torch.tensor(gtA).unsqueeze(0).unsqueeze(0), high_resol_coef)[0][0].detach().cpu().numpy()
    pde_map_gtb = burgers_pde_all_pixel(torch.tensor(gtB).unsqueeze(0).unsqueeze(0), low_resol_coef)[0][0].detach().cpu().numpy()
    pde_map_predict = burgers_pde_all_pixel(predict.unsqueeze(0).unsqueeze(0) , low_resol_coef)[0][0].detach().cpu().numpy()
    pde_map = np.abs(pde_map_predict - pde_map_gtb)
    pde_map_predict_initial = burgers_pde_all_pixel(torch.tensor(dataB_predict_initial).unsqueeze(0).unsqueeze(0), estimated_coef)[0][0].detach().cpu().numpy()
    
    finite_err = err_after[np.isfinite(err_after)]
    vmin_e = 0.0
    vmax_e = np.percentile(finite_err, 99) if finite_err.size > 0 else 1.0
    
    finite_pde_gta = pde_map_gta[1:-1, 1:-1][np.isfinite(pde_map_gta[1:-1, 1:-1])]
    vmin_gta = 0.0
    vmax_gta = np.percentile(finite_pde_gta, 99) if finite_pde_gta.size > 0 else 1.0 
    
    finite_pde_gtb = pde_map_gtb[1:-1, 1:-1][np.isfinite(pde_map_gtb[1:-1, 1:-1])]
    vmin_gtb = 0.0
    vmax_gtb = np.percentile(finite_pde_gtb, 99) if finite_pde_gtb.size > 0 else 1.0 
    
    finite_pde_predict = pde_map_predict[1:-1, 1:-1][np.isfinite(pde_map_predict[1:-1, 1:-1])]
    vmin_predict = 0.0
    vmax_predict = np.percentile(finite_pde_predict, 99) if finite_pde_predict.size > 0 else 1.0 
    
    finite_pde_predict_initial = pde_map_predict_initial[1:-1, 1:-1][np.isfinite(pde_map_predict_initial[1:-1, 1:-1])]
    vmin_predict_initial = 0.0
    vmax_predict_initial = np.percentile(finite_pde_predict_initial, 99) if finite_pde_predict_initial.size > 0 else 1.0 
    
    if mask_1d is not None:
        mask = np.asarray(mask_1d, dtype=bool).reshape(-1)
        err_after  = err_after.copy()
        err_after[:,  ~mask] = np.nan
        pde_map  = pde_map.copy()
        pde_map[:,  ~mask] = np.nan


    
    cmap_err = plt.cm.get_cmap(cmap).copy()
    cmap_err.set_bad(color="white")

    fig, axes = plt.subplots(2, 5, figsize=(15, 5))

    im_gtA = axes[0, 0].imshow(
        gtA, origin="upper", extent=extent,
        aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
    )
    axes[0, 0].set_title(f"GT A_{np.round(high_resol_coef, 4)}", fontsize=10)

    im_gtB = axes[0, 1].imshow(
        gtB, origin="upper", extent=extent,
        aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
    )
    axes[0, 1].set_title(f"GT B_{np.round(low_resol_coef, 4)}", fontsize=10)

    im_generated = axes[0, 2].imshow(
        gen_after, origin="upper", extent=extent,
        aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
    )
    axes[0, 2].set_title("Generated B", fontsize=10)

    im_predict_initial = axes[0, 3].imshow(
        dataB_predict_initial, origin="upper", extent=extent,
        aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
    )
    axes[0, 3].set_title(f"Feasible?_{estimated_coef.detach().cpu().numpy()}", fontsize=10)
    
    im_err_after = axes[0, 4].imshow(
        err_after, origin="upper", extent=extent,
        aspect="equal", cmap=cmap_err, vmin=vmin_e, vmax=vmax_e,
        interpolation="nearest"
    )
    axes[0, 4].set_title("Masked error", fontsize=10)
    axes[0, 4].set_facecolor("white")
        
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
    
    im_pde_predict_initial = axes[1, 3].imshow(
        pde_map_predict_initial, origin="upper", extent=extent,
        aspect="equal", cmap=cmap_err, vmin=vmin_predict_initial, vmax=vmax_predict_initial,
        interpolation="nearest"
    )
    axes[1, 3].set_title("Feasible?_PDE error", fontsize=10)
    axes[1, 3].set_facecolor("white")
    
    im_pde_err = axes[1, 4].imshow(
        pde_map, origin="upper", extent=extent,
        aspect="equal", cmap=cmap_err, vmin=vmin_predict, vmax=vmax_predict,
        interpolation="nearest"
    )
    axes[1, 4].set_title("PDE error GT vs Generated", fontsize=10)
    axes[1, 4].set_facecolor("white")
    # for ax in axes:
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("t")
    div0 = make_axes_locatable(axes[0, 0])
    cax_gt = div0.append_axes("right", size="6%", pad=0.08)
    cb_gt = fig.colorbar(im_gtA, cax=cax_gt)
    cb_gt.set_label("Field value", fontsize=10)
    
    div1 = make_axes_locatable(axes[0, 1])
    cax_gt = div1.append_axes("right", size="6%", pad=0.08)
    cb_gt = fig.colorbar(im_gtB, cax=cax_gt)
    cb_gt.set_label("Field value", fontsize=10)
    
    div2 = make_axes_locatable(axes[0, 2])
    cax_gt = div2.append_axes("right", size="6%", pad=0.08)
    cb_gt = fig.colorbar(im_generated, cax=cax_gt)
    cb_gt.set_label("Field value", fontsize=10)

    div3 = make_axes_locatable(axes[0, 3])
    cax_predict_initial = div3.append_axes("right", size="6%", pad=0.08)
    cb_predict_initial = fig.colorbar(im_predict_initial, cax=cax_predict_initial)
    cb_predict_initial.set_label("Field value", fontsize=10)
    
    div4 = make_axes_locatable(axes[0, 4])
    cax_err = div4.append_axes("right", size="6%", pad=0.08)
    cb_err = fig.colorbar(im_err_after, cax=cax_err)
    cb_err.set_label("Masked error", fontsize=10)

    div5 = make_axes_locatable(axes[1, 0])
    cax_pde = div5.append_axes("right", size="6%", pad=0.08)
    cb_pde = fig.colorbar(im_pde_gta, cax=cax_pde)
    cb_pde.set_label("PDE residual", fontsize=10)
    
    div6 = make_axes_locatable(axes[1, 1])
    cax_pde = div6.append_axes("right", size="6%", pad=0.08)
    cb_pde = fig.colorbar(im_pde_gtb, cax=cax_pde)
    cb_pde.set_label("PDE residual", fontsize=10)
    
    div7 = make_axes_locatable(axes[1, 2])
    cax_pde = div7.append_axes("right", size="6%", pad=0.08)
    cb_pde = fig.colorbar(im_pde_predict, cax=cax_pde)
    cb_pde.set_label("PDE residual", fontsize=10)
    
    div8 = make_axes_locatable(axes[1, 3])
    cax_pde_predict_initial = div8.append_axes("right", size="6%", pad=0.08)
    cb_pde_predict_initial = fig.colorbar(im_pde_predict_initial, cax=cax_pde_predict_initial)
    cb_pde_predict_initial.set_label("PDE residual", fontsize=10)
    
    div9 = make_axes_locatable(axes[1, 4])
    cax_pde = div9.append_axes("right", size="6%", pad=0.08)
    cb_pde = fig.colorbar(im_pde_err, cax=cax_pde)
    cb_pde.set_label("PDE error", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 1])
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