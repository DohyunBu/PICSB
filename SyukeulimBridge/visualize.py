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
    ref = data[0]
    vmin = np.nanmin(ref)
    vmax = np.nanmax(ref)
    if C == 1:
        full = data[batch_idx, 0]    
        m = mask_bool[batch_idx, 0]        

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

def plot_result(
    dataA, dataB, dataB_predict_initial,
    predict,
    cmap="turbo",
    high_resol_coef=None,
    low_resol_coef=None,
    estimated_coef=None,
    mask_2d=None,
):
    if isinstance(predict, (list, tuple)):
        pred_path = predict
        pred_last = pred_path[-1]
    else:
        pred_path = None
        pred_last = predict

    gtA = dataA[0, 0].detach().cpu().numpy() if torch.is_tensor(dataA) else np.asarray(dataA)[0, 0]
    gtB = dataB[0, 0].detach().cpu().numpy() if torch.is_tensor(dataB) else np.asarray(dataB)[0, 0]
    generated_B = predict[0, 0].detach().cpu().numpy() if torch.is_tensor(predict) else np.asarray(predict)[0, 0]
    dataB_predict_initial = dataB_predict_initial[0, 0].detach().cpu().numpy() if torch.is_tensor(dataB_predict_initial) else np.asarray(dataB_predict_initial)[0, 0]
    
    all_vals = np.concatenate([gtA.ravel(), gtB.ravel()])
    vmin_gt, vmax_gt = np.nanmin(all_vals), np.nanmax(all_vals)

    err_after = np.abs(gtB - generated_B)

    if mask_2d is not None:
        m = mask_2d
        if torch.is_tensor(m):
            m = m.detach().cpu().numpy()
        m = np.asarray(m, dtype=bool)
        m0 = m[0, 0]
        err_after = err_after.copy()
        err_after[~m0] = np.nan
    else:
        m0 = None

    finite_err = err_after[np.isfinite(err_after)]
    vmin_e = 0.0
    vmax_e = np.percentile(finite_err, 99) if finite_err.size > 0 else 1.0

    def _pde_map(u_tx, coef):
            out = loss.pde_loss_visualization(
                torch.tensor(u_tx).unsqueeze(0).unsqueeze(0).float(), torch.tensor(coef)
            )
            return out[0, 0].detach().cpu().numpy()

    pde_map_gta = _pde_map(gtA, low_resol_coef)
    pde_map_gtb = _pde_map(gtB, high_resol_coef)
    pde_map_pred = _pde_map(generated_B, high_resol_coef)
    pde_map_init = _pde_map(dataB_predict_initial, estimated_coef)

    def _robust_vmax(pde):
        if pde is None:
            return (0.0, 1.0)
        finite = pde[np.isfinite(pde)]
        if finite.size == 0:
            return (0.0, 1.0)
        return (0.0, np.percentile(finite, 99))

    vmin_gta, vmax_gta = _robust_vmax(pde_map_gta)
    vmin_gtb, vmax_gtb = _robust_vmax(pde_map_gtb)
    vmin_pred, vmax_pred = _robust_vmax(pde_map_pred)
    vmin_init, vmax_init = _robust_vmax(pde_map_init)

    cmap_err = plt.cm.get_cmap(cmap).copy()
    cmap_err.set_bad(color="white")

    fig, axes = plt.subplots(2, 5, figsize=(18, 6), constrained_layout=True)

    im_gtA = axes[0, 0].imshow(gtA, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 0].set_title(f"GT Low fidelity (coef={low_resol_coef})", fontsize=10)

    im_gtB = axes[0, 1].imshow(gtB, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 1].set_title(f"GT High fidelity (coef={high_resol_coef})", fontsize=10)

    im_generated = axes[0, 2].imshow(generated_B, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 2].set_title("Generated High fidelity", fontsize=10)

    im_init = axes[0, 3].imshow(dataB_predict_initial, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 3].set_title(f"GT with estimated coef (coef={estimated_coef})", fontsize=10)

    im_err = axes[0, 4].imshow(err_after, origin="upper", aspect="equal", cmap=cmap_err, vmin=vmin_e, vmax=vmax_e, interpolation="nearest")
    axes[0, 4].set_title("Masked error |GT-Gen|", fontsize=10)
    axes[0, 4].set_facecolor("white")

    def _imshow_or_blank(ax, arr, title, vmin, vmax):
        if arr is None:
            ax.axis("off")
            ax.set_title(title + " (N/A)", fontsize=10)
            return None
        im = ax.imshow(arr, origin="upper", aspect="equal", cmap=cmap_err, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_facecolor("white")
        return im

    im_pde_gta = _imshow_or_blank(axes[1, 0], pde_map_gta, "GT Low PDE residual map", vmin_gta, vmax_gta)
    im_pde_gtb = _imshow_or_blank(axes[1, 1], pde_map_gtb, "GT high PDE residual map", vmin_gtb, vmax_gtb)
    im_pde_pred = _imshow_or_blank(axes[1, 2], pde_map_pred, "Generated PDE residual map", vmin_pred, vmax_pred)
    im_pde_init = _imshow_or_blank(axes[1, 3], pde_map_init, "Feasible?", vmin_init, vmax_init)

    pde_diff = np.abs(pde_map_pred - pde_map_gtb)
    if m0 is not None:
        pde_diff = pde_diff.copy()
        pde_diff[~m0] = np.nan
    im_pde_diff = axes[1, 4].imshow(pde_diff, origin="upper", aspect="equal", cmap=cmap_err,
                                    vmin=vmin_pred, vmax=vmax_pred, interpolation="nearest")
    axes[1, 4].set_title("PDE |Gen - GT|", fontsize=10)
    axes[1, 4].set_facecolor("white")

    def _add_cb(ax, im, label):
        if im is None:
            return
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="6%", pad=0.08)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=9)

    _add_cb(axes[0, 0], im_gtA, "Field value")
    _add_cb(axes[0, 1], im_gtB, "Field value")
    _add_cb(axes[0, 2], im_generated, "Field value")
    _add_cb(axes[0, 3], im_init, "Field value")
    _add_cb(axes[0, 4], im_err, "Masked error")

    _add_cb(axes[1, 0], im_pde_gta, "PDE residual")
    _add_cb(axes[1, 1], im_pde_gtb, "PDE residual")
    _add_cb(axes[1, 2], im_pde_pred, "PDE residual")
    _add_cb(axes[1, 3], im_pde_init, "PDE residual")
    _add_cb(axes[1, 4], im_pde_diff, "PDE error")

    plt.show()
