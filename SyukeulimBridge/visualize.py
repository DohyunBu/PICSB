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
    dataA, dataB,
    predict, filled,
    cmap="turbo",
    high_resol_coef=None,
    low_resol_coef=None,
    mask_2d=None
):
    
    gtA = dataA[0, 0].detach().cpu().numpy() if torch.is_tensor(dataA) else np.asarray(dataA)[0, 0]
    gtB = dataB[0, 0].detach().cpu().numpy() if torch.is_tensor(dataB) else np.asarray(dataB)[0, 0]
    generated_B = predict[0, 0].detach().cpu().numpy() if torch.is_tensor(predict) else np.asarray(predict)[0, 0]
    filled = filled[0, 0].detach().cpu().numpy() if torch.is_tensor(filled) else np.asarray(filled)[0, 0]
    
    all_vals = np.concatenate([gtA.ravel(), gtB.ravel()])
    vmin_gt, vmax_gt = np.nanmin(all_vals), np.nanmax(all_vals)

    err_before = np.abs(gtB - gtA)
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

    all_vals_err = np.concatenate([err_after.ravel()])
    vmin_e, vmax_e = np.nanmin(all_vals_err), np.nanmax(all_vals_err)

    def _pde_map(u_tx, coef):
            out = loss.burgers_visualization(
                torch.tensor(u_tx).unsqueeze(0).unsqueeze(0).float(), torch.tensor(coef)
            )
            return out[0, 0].detach().cpu().numpy()

    pde_map_gta = _pde_map(gtA, low_resol_coef)
    pde_map_gtb = _pde_map(gtB, high_resol_coef)
    pde_map_pred = _pde_map(generated_B, high_resol_coef)
    pde_map_filled = _pde_map(filled, high_resol_coef)
    
    all_vals_pde = np.concatenate([pde_map_gtb[1:-1, 2:-2].ravel()])
    vmin_pde, vmax_pde = np.nanmin(all_vals_pde), np.nanmax(all_vals_pde)

    all_vals_pde_error = np.concatenate([(np.abs(pde_map_gtb[1:-1, 2:-2] - pde_map_pred[1:-1, 2:-2])).ravel()])
    vmin_pde_err, vmax_pde_err = np.nanmin(all_vals_pde_error), np.nanmax(all_vals_pde_error)
    
    cmap_err = plt.cm.get_cmap(cmap).copy()
    cmap_err.set_bad(color="white")

    fig, axes = plt.subplots(2, 5, figsize=(18, 6), constrained_layout=True)
    
    
    im_gtA = axes[0, 0].imshow(gtA, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 0].set_title(f"GT Low fidelity (coef={low_resol_coef})", fontsize=10)

    im_gtB = axes[0, 1].imshow(gtB, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 1].set_title(f"GT High fidelity (coef={high_resol_coef})", fontsize=10)

    im_generated = axes[0, 2].imshow(generated_B, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 2].set_title("Generated High fidelity", fontsize=10)

    im_err_before = axes[0, 3].imshow(err_before, origin="upper", aspect="equal", cmap=cmap_err, vmin=vmin_e, vmax=vmax_e)
    axes[0, 3].set_title(f"Masked error |GT_high-GT_low| {err_before.mean():.4f}", fontsize=10)

    im_err = axes[0, 4].imshow(err_after, origin="upper", aspect="equal", cmap=cmap_err, vmin=vmin_e, vmax=vmax_e, interpolation="nearest")
    axes[0, 4].set_title(f"Masked error |GT_high-Gen| {err_after.mean():.4f}", fontsize=10)
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
    pde_result_gta = np.round(pde_map_gta[1:-1, 2:-2].mean(), 4)
    im_pde_gta = _imshow_or_blank(axes[1, 0], pde_map_gta, f"GT Low PDE residual map ({pde_result_gta:.4f})", vmin_pde, vmax_pde)
    
    pde_result_gtb = np.round(pde_map_gtb[1:-1, 2:-2].mean(), 4)
    im_pde_gtb = _imshow_or_blank(axes[1, 1], pde_map_gtb, f"GT high PDE residual map ({pde_result_gtb:.4f})", vmin_pde, vmax_pde)

    pde_result_pred = np.round(pde_map_pred[1:-1, 2:-2].mean(), 4)
    im_pde_pred = _imshow_or_blank(axes[1, 2], pde_map_pred, f"Gen PDE residual map ({pde_result_pred:.4f})", vmin_pde, vmax_pde)
    
    pde_diff_filled = np.abs(pde_map_filled - pde_map_gtb)
    im_pde_filled = axes[1, 3].imshow(pde_diff_filled, origin="upper", aspect="equal", cmap=cmap_err,
                                    vmin=vmin_pde_err, vmax=vmax_pde_err, interpolation="nearest")
    axes[1, 3].set_title(f"PDE error |GT_high - Interpolated| {pde_diff_filled[1:-1, 2:-2].mean():.4f}", fontsize=10)
    axes[1, 3].set_facecolor("white")
    
    pde_diff = np.abs(pde_map_pred - pde_map_gtb)
    if m0 is not None:
        pde_diff = pde_diff.copy()
        pde_diff[~m0] = np.nan
    im_pde_diff = axes[1, 4].imshow(pde_diff, origin="upper", aspect="equal", cmap=cmap_err,
                                    vmin=vmin_pde_err, vmax=vmax_pde_err, interpolation="nearest")
    axes[1, 4].set_title(f"PDE error |GT_high - Gen| {pde_diff[1:-1, 2:-2].mean():.4f}", fontsize=10)
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
    _add_cb(axes[0, 3], im_err_before, "Masked error")
    _add_cb(axes[0, 4], im_err, "Masked error")

    _add_cb(axes[1, 0], im_pde_gta, "PDE residual")
    _add_cb(axes[1, 1], im_pde_gtb, "PDE residual")
    _add_cb(axes[1, 2], im_pde_pred, "PDE residual")
    _add_cb(axes[1, 3], im_pde_filled, "PDE residual")
    _add_cb(axes[1, 4], im_pde_diff, "PDE error")

    plt.show()

def plot_result_3d(
    dataA, dataB,
    predict,
    cmap="turbo",
    high_resol_coef=None,
    low_resol_coef=None,
    mask=None,
    snapshot_idx=None,
):
    
    gtA = dataA[[0], :].detach().cpu().numpy() if torch.is_tensor(dataA) else np.asarray(dataA)[0, 0]
    gtB = dataB[[0], :].detach().cpu().numpy() if torch.is_tensor(dataB) else np.asarray(dataB)[0, 0]
    generated_B = predict[[0], :].detach().cpu().numpy() if torch.is_tensor(predict) else np.asarray(predict)[0, 0]

    gtA_snapshot = gtA[0, snapshot_idx, :, :]
    gtB_snapshot = gtB[0, snapshot_idx, :, :]
    generated_B_snapshot = generated_B[0, snapshot_idx, :, :]
    
    all_vals = np.concatenate([gtA_snapshot.ravel(), gtB_snapshot.ravel()])
    vmin_gt, vmax_gt = np.nanmin(all_vals), np.nanmax(all_vals)

    err_before = np.abs(gtB - gtA)
    err_after = np.abs(gtB - generated_B)

    if mask is not None:
        m = mask
        if torch.is_tensor(m):
            m = m.detach().cpu().numpy()
        m = np.asarray(m, dtype=bool)
        m0 = m
        err_after = err_after.copy()
        err_after[~m0] = np.nan
    else:
        m0 = None
        
    err_before_snapshot = err_before[0, snapshot_idx, :, :]
    err_after_snapshot = err_after[0, snapshot_idx, :, :]
    
    all_vals_err = np.concatenate([err_after.ravel()])
    vmin_e, vmax_e = np.nanmin(all_vals_err), np.nanmax(all_vals_err)

    def _pde_map(u_tx, coef):
            out = loss.kol_visualization(
                torch.tensor(u_tx).float(), torch.tensor(coef)
            )
            return out.detach().cpu().numpy()
    pde_map_gta = _pde_map(dataA, low_resol_coef)
    pde_map_gtb = _pde_map(dataB, high_resol_coef)
    pde_map_pred = _pde_map(predict, high_resol_coef)
    
    pde_map_gta_snapshot = pde_map_gta[0, snapshot_idx, :, :]
    pde_map_gtb_snapshot = pde_map_gtb[0, snapshot_idx, :, :]
    pde_map_pred_snapshot = pde_map_pred[0, snapshot_idx, :, :]
    

    all_vals_pde = np.concatenate([pde_map_gtb_snapshot.ravel()])
    vmin_pde, vmax_pde = np.nanmin(all_vals_pde), np.nanmax(all_vals_pde)

    all_vals_pde_error = np.concatenate([(np.abs(pde_map_gtb_snapshot - pde_map_pred_snapshot)).ravel()])
    vmin_pde_err, vmax_pde_err = np.nanmin(all_vals_pde_error), np.nanmax(all_vals_pde_error)
    
    cmap_err = plt.cm.get_cmap(cmap).copy()
    cmap_err.set_bad(color="white")

    fig, axes = plt.subplots(2, 5, figsize=(18, 6), constrained_layout=True)
    axes[1, 3].set_visible(False)
    
    im_gtA = axes[0, 0].imshow(gtA_snapshot, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 0].set_title("GT Low fidelity)", fontsize=10)

    im_gtB = axes[0, 1].imshow(gtB_snapshot, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 1].set_title("GT High fidelity)", fontsize=10)

    im_generated = axes[0, 2].imshow(generated_B_snapshot, origin="upper", aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 2].set_title("Generated High fidelity", fontsize=10)

    im_err_before = axes[0, 3].imshow(err_before_snapshot, origin="upper", aspect="equal", cmap=cmap_err, vmin=vmin_e, vmax=vmax_e)
    axes[0, 3].set_title(f"Masked error |GT_high-GT_low| {np.abs(gtB - gtA).mean():.4f}", fontsize=10)

    im_err = axes[0, 4].imshow(err_after_snapshot, origin="upper", aspect="equal", cmap=cmap_err, vmin=vmin_e, vmax=vmax_e, interpolation="nearest")
    axes[0, 4].set_title(f"Masked error |GT_high-Gen| {np.abs(gtB - generated_B).mean():.4f}", fontsize=10)
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
    
    pde_result_gta = np.round(pde_map_gta.mean(), 4)
    im_pde_gta = _imshow_or_blank(axes[1, 0], pde_map_gta_snapshot, f"GT Low PDE residual map ({pde_result_gta:.4f})", vmin_pde, vmax_pde)
    
    pde_result_gtb = np.round(pde_map_gtb.mean(), 4)
    im_pde_gtb = _imshow_or_blank(axes[1, 1], pde_map_gtb_snapshot, f"GT high PDE residual map ({pde_result_gtb:.4f})", vmin_pde, vmax_pde)

    pde_result_pred = np.round(pde_map_pred.mean(), 4)
    im_pde_pred = _imshow_or_blank(axes[1, 2], pde_map_pred_snapshot, f"Gen PDE residual map ({pde_result_pred:.4f})", vmin_pde, vmax_pde)
    
    pde_diff = np.abs(pde_map_pred - pde_map_gtb)
    if m0 is not None:
        pde_diff = pde_diff.copy()
        pde_diff[~(m0[:,1:-1,:,:])] = np.nan
    pde_diff_snapshot = pde_diff[0, snapshot_idx, :, :]
    im_pde_diff = axes[1, 4].imshow(pde_diff_snapshot, origin="upper", aspect="equal", cmap=cmap_err,
                                    vmin=vmin_pde_err, vmax=vmax_pde_err, interpolation="nearest")
    axes[1, 4].set_title(f"PDE error |GT_high - Gen| {np.abs(pde_map_pred - pde_map_gtb).mean():.4f}", fontsize=10)
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
    _add_cb(axes[0, 3], im_err_before, "Masked error")
    _add_cb(axes[0, 4], im_err, "Masked error")

    _add_cb(axes[1, 0], im_pde_gta, "PDE residual")
    _add_cb(axes[1, 1], im_pde_gtb, "PDE residual")
    _add_cb(axes[1, 2], im_pde_pred, "PDE residual")
    _add_cb(axes[1, 4], im_pde_diff, "PDE error")

    plt.show()
