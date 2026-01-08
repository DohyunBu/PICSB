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
    
# def plot_result(
#     dataA, dataB, dataB_predict_initial,
#     predict,
#     cmap="turbo",
#     high_resol_coef=None,
#     low_resol_coef=None,
#     estimated_coef=None,
#     mask_2d=None,
# ):
    
#     after = (
#         predict.detach().cpu().numpy()
#         if torch.is_tensor(predict)
#         else np.asarray(predict)
#     )

#     gen_after  = after[0]  if after.ndim == 3  else after

#     gtA = dataA[0][0]
#     gtB = dataB[0][0]
#     dataB_predict_initial = dataB_predict_initial[0][0]
    
#     gtA = np.asarray(gtA)
#     gtB = np.asarray(gtB)
#     dataB_predict_initial = np.asarray(dataB_predict_initial)
#     gen_after  = np.asarray(gen_after)
        

#     extent = [0, 1, 1, 0]

#     all_vals = np.concatenate([gtA.ravel(), gtB.ravel()])
#     vmin_gt, vmax_gt = np.nanmin(all_vals), np.nanmax(all_vals)

#     err_after  = np.abs(gtB - gen_after)
    
#     pde_map_gta = loss.pde_loss_visualization(torch.tensor(gtA).unsqueeze(0).unsqueeze(0), high_resol_coef)[0][0].detach().cpu().numpy()
#     pde_map_gtb = loss.pde_loss_visualization(torch.tensor(gtB).unsqueeze(0).unsqueeze(0), low_resol_coef)[0][0].detach().cpu().numpy()
#     pde_map_predict = loss.pde_loss_visualization(predict.unsqueeze(0).unsqueeze(0) , low_resol_coef)[0][0].detach().cpu().numpy()
#     pde_map = np.abs(pde_map_predict - pde_map_gtb)
#     pde_map_predict_initial = loss.pde_loss_visualization(torch.tensor(dataB_predict_initial).unsqueeze(0).unsqueeze(0), torch.exp(estimated_coef))[0][0].detach().cpu().numpy()
    
#     finite_err = err_after[np.isfinite(err_after)]
#     vmin_e = 0.0
#     vmax_e = np.percentile(finite_err, 99) if finite_err.size > 0 else 1.0
    
#     finite_pde_gta = pde_map_gta[1:-1, 2:-2][np.isfinite(pde_map_gta[1:-1, 2:-2])]
#     vmin_gta = 0.0
#     vmax_gta = np.percentile(finite_pde_gta, 99) if finite_pde_gta.size > 0 else 1.0 
    
#     finite_pde_gtb = pde_map_gtb[1:-1, 2:-2][np.isfinite(pde_map_gtb[1:-1, 2:-2])]
#     vmin_gtb = 0.0
#     vmax_gtb = np.percentile(finite_pde_gtb, 99) if finite_pde_gtb.size > 0 else 1.0 
    
#     finite_pde_predict = pde_map_predict[1:-1, 2:-2][np.isfinite(pde_map_predict[1:-1, 2:-2])]
#     vmin_predict = 0.0
#     vmax_predict = np.percentile(finite_pde_predict, 99) if finite_pde_predict.size > 0 else 1.0 
    
#     finite_pde_predict_initial = pde_map_predict_initial[1:-1, 2:-2][np.isfinite(pde_map_predict_initial[1:-1, 2:-2])]
#     vmin_predict_initial = 0.0
#     vmax_predict_initial = np.percentile(finite_pde_predict_initial, 99) if finite_pde_predict_initial.size > 0 else 1.0 
    
#     if mask_2d is not None:
#         mask = np.asarray(mask_2d, dtype=bool)
#         err_after  = err_after.copy()
#         err_after[~mask] = np.nan
#         pde_map  = pde_map.copy()
#         pde_map[~mask] = np.nan


    
#     cmap_err = plt.cm.get_cmap(cmap).copy()
#     cmap_err.set_bad(color="white")

#     fig, axes = plt.subplots(2, 5, figsize=(15, 5))

#     im_gtA = axes[0, 0].imshow(
#         gtA, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
#     )
#     axes[0, 0].set_title(f"GT A_{np.round(high_resol_coef, 4)}", fontsize=10)

#     im_gtB = axes[0, 1].imshow(
#         gtB, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
#     )
#     axes[0, 1].set_title(f"GT B_{np.round(low_resol_coef, 4)}", fontsize=10)

#     im_generated = axes[0, 2].imshow(
#         gen_after, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
#     )
#     axes[0, 2].set_title("Generated B", fontsize=10)

#     im_predict_initial = axes[0, 3].imshow(
#         dataB_predict_initial, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt
#     )
#     axes[0, 3].set_title(f"Feasible?_{np.exp(estimated_coef.detach().cpu().numpy())}", fontsize=10)
    
#     im_err_after = axes[0, 4].imshow(
#         err_after, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap_err, vmin=vmin_e, vmax=vmax_e,
#         interpolation="nearest"
#     )
#     axes[0, 4].set_title("Masked error", fontsize=10)
#     axes[0, 4].set_facecolor("white")
        
#     im_pde_gta = axes[1, 0].imshow(
#         pde_map_gta, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap_err, vmin=vmin_gta, vmax=vmax_gta,
#         interpolation="nearest"
#     )
#     axes[1, 0].set_title("GT A PDE error", fontsize=10)
#     axes[1, 0].set_facecolor("white")
    
#     im_pde_gtb = axes[1, 1].imshow(
#         pde_map_gtb, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap_err, vmin=vmin_gtb, vmax=vmax_gtb,
#         interpolation="nearest"
#     )
#     axes[1, 1].set_title("GT B PDE error", fontsize=10)
#     axes[1, 1].set_facecolor("white")
    
#     im_pde_predict = axes[1, 2].imshow(
#         pde_map_predict, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap_err, vmin=vmin_predict, vmax=vmax_predict,
#         interpolation="nearest"
#     )
#     axes[1, 2].set_title("Generated B PDE error", fontsize=10)
#     axes[1, 2].set_facecolor("white")
    
#     im_pde_predict_initial = axes[1, 3].imshow(
#         pde_map_predict_initial, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap_err, vmin=vmin_predict_initial, vmax=vmax_predict_initial,
#         interpolation="nearest"
#     )
#     axes[1, 3].set_title("Feasible?_PDE error", fontsize=10)
#     axes[1, 3].set_facecolor("white")
    
#     im_pde_err = axes[1, 4].imshow(
#         pde_map, origin="upper", extent=extent,
#         aspect="equal", cmap=cmap_err, vmin=vmin_predict, vmax=vmax_predict,
#         interpolation="nearest"
#     )
#     axes[1, 4].set_title("PDE error GT vs Generated", fontsize=10)
#     axes[1, 4].set_facecolor("white")
#     # for ax in axes:
#     #     ax.set_xlabel("x")
#     #     ax.set_ylabel("t")
#     div0 = make_axes_locatable(axes[0, 0])
#     cax_gt = div0.append_axes("right", size="6%", pad=0.08)
#     cb_gt = fig.colorbar(im_gtA, cax=cax_gt)
#     cb_gt.set_label("Field value", fontsize=10)
    
#     div1 = make_axes_locatable(axes[0, 1])
#     cax_gt = div1.append_axes("right", size="6%", pad=0.08)
#     cb_gt = fig.colorbar(im_gtB, cax=cax_gt)
#     cb_gt.set_label("Field value", fontsize=10)
    
#     div2 = make_axes_locatable(axes[0, 2])
#     cax_gt = div2.append_axes("right", size="6%", pad=0.08)
#     cb_gt = fig.colorbar(im_generated, cax=cax_gt)
#     cb_gt.set_label("Field value", fontsize=10)

#     div3 = make_axes_locatable(axes[0, 3])
#     cax_predict_initial = div3.append_axes("right", size="6%", pad=0.08)
#     cb_predict_initial = fig.colorbar(im_predict_initial, cax=cax_predict_initial)
#     cb_predict_initial.set_label("Field value", fontsize=10)
    
#     div4 = make_axes_locatable(axes[0, 4])
#     cax_err = div4.append_axes("right", size="6%", pad=0.08)
#     cb_err = fig.colorbar(im_err_after, cax=cax_err)
#     cb_err.set_label("Masked error", fontsize=10)

#     div5 = make_axes_locatable(axes[1, 0])
#     cax_pde = div5.append_axes("right", size="6%", pad=0.08)
#     cb_pde = fig.colorbar(im_pde_gta, cax=cax_pde)
#     cb_pde.set_label("PDE residual", fontsize=10)
    
#     div6 = make_axes_locatable(axes[1, 1])
#     cax_pde = div6.append_axes("right", size="6%", pad=0.08)
#     cb_pde = fig.colorbar(im_pde_gtb, cax=cax_pde)
#     cb_pde.set_label("PDE residual", fontsize=10)
    
#     div7 = make_axes_locatable(axes[1, 2])
#     cax_pde = div7.append_axes("right", size="6%", pad=0.08)
#     cb_pde = fig.colorbar(im_pde_predict, cax=cax_pde)
#     cb_pde.set_label("PDE residual", fontsize=10)
    
#     div8 = make_axes_locatable(axes[1, 3])
#     cax_pde_predict_initial = div8.append_axes("right", size="6%", pad=0.08)
#     cb_pde_predict_initial = fig.colorbar(im_pde_predict_initial, cax=cax_pde_predict_initial)
#     cb_pde_predict_initial.set_label("PDE residual", fontsize=10)
    
#     div9 = make_axes_locatable(axes[1, 4])
#     cax_pde = div9.append_axes("right", size="6%", pad=0.08)
#     cb_pde = fig.colorbar(im_pde_err, cax=cax_pde)
#     cb_pde.set_label("PDE error", fontsize=10)
#     plt.tight_layout(rect=[0, 0, 1, 1])
#     plt.show()
    
# def stitch_windows(gen_vectors, starts, win_T=128, Nx=128, stride=128):
#     nW = len(starts)
#     Nt = starts[-1] + win_T
#     full = torch.zeros(Nt, Nx)
#     for i in range(nW):
#         block = gen_vectors[i].reshape(win_T, Nx)
#         start = starts[i]
#         full[start:start+win_T] = block
#     return full
def plot_result(
    dataA, dataB, dataB_predict_initial,
    predict,
    cmap="turbo",
    high_resol_coef=None,
    low_resol_coef=None,
    estimated_coef=None,
    mask_2d=None,
):
    # -------------------------
    # 0) predict 경로 처리
    # predict: list[tensor] 또는 tensor
    # -------------------------
    if isinstance(predict, (list, tuple)):
        pred_path = predict
        pred_last = pred_path[-1]
        pred_init = pred_path[0]
    else:
        pred_path = None
        pred_last = predict
        pred_init = None

    gtA = dataA[0, 0].detach().cpu().numpy() if torch.is_tensor(dataA) else np.asarray(dataA)[0, 0]
    gtB = dataB[0, 0].detach().cpu().numpy() if torch.is_tensor(dataB) else np.asarray(dataB)[0, 0]

    if dataB_predict_initial is None:
        if pred_init is None:
            dataB_predict_initial = gtA
        else:
            dataB_predict_initial = pred_init[0, 0].detach().cpu().numpy()
    else:
        if torch.is_tensor(dataB_predict_initial):
            dataB_predict_initial = dataB_predict_initial[0, 0].detach().cpu().numpy()
        else:
            dataB_predict_initial = np.asarray(dataB_predict_initial)[0, 0]

    if torch.is_tensor(pred_last):
        gen_after = pred_last[0, 0].detach().cpu().numpy()
    else:
        gen_after = np.asarray(pred_last)[0, 0]

    # -------------------------
    # 2) vmin/vmax (GT 기준 고정)
    # -------------------------
    all_vals = np.concatenate([gtA.ravel(), gtB.ravel()])
    vmin_gt, vmax_gt = np.nanmin(all_vals), np.nanmax(all_vals)

    # -------------------------
    # 3) error map (+ mask 적용)
    # -------------------------
    err_after = np.abs(gtB - gen_after)

    if mask_2d is not None:
        m = mask_2d
        if torch.is_tensor(m):
            m = m.detach().cpu().numpy()
        m = np.asarray(m, dtype=bool)
        # (B,1,T,X) 가정
        m0 = m[0, 0]
        err_after = err_after.copy()
        err_after[~m0] = np.nan
    else:
        m0 = None

    finite_err = err_after[np.isfinite(err_after)]
    vmin_e = 0.0
    vmax_e = np.percentile(finite_err, 99) if finite_err.size > 0 else 1.0

    # -------------------------
    # 4) PDE residual map (Burgers)
    #    loss.pde_loss_visualization가 네 코드에 실제로 있다면 그대로 쓰고,
    #    없다면 아래 try/except로 안전 처리
    # -------------------------
    def _pde_map(u_tx, coef):
        # u_tx: (T,X) numpy
        try:
            # 네 코드에서 쓰던 함수
            out = loss.pde_loss_visualization(
                torch.tensor(u_tx).unsqueeze(0).unsqueeze(0).float(), coef
            )
            return out[0, 0].detach().cpu().numpy()
        except Exception:
            # fallback: map 없이 전체 residual만 표시하고 싶으면 여기서 구현 필요
            return None

    pde_map_gta = _pde_map(gtA, high_resol_coef) if high_resol_coef is not None else None
    pde_map_gtb = _pde_map(gtB, low_resol_coef)  if low_resol_coef  is not None else None
    pde_map_pred = _pde_map(gen_after, low_resol_coef) if low_resol_coef is not None else None
    pde_map_init = _pde_map(dataB_predict_initial, low_resol_coef) if low_resol_coef is not None else None

    # PDE vmin/vmax는 각자 robust하게
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

    # -------------------------
    # 5) plotting
    # -------------------------
    cmap_err = plt.cm.get_cmap(cmap).copy()
    cmap_err.set_bad(color="white")

    fig, axes = plt.subplots(2, 5, figsize=(18, 6), constrained_layout=True)

    # Row 0: fields + masked error
    im_gtA = axes[0, 0].imshow(gtA, origin="upper", aspect="auto", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 0].set_title(f"GT A (nu={high_resol_coef})", fontsize=10)

    im_gtB = axes[0, 1].imshow(gtB, origin="upper", aspect="auto", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 1].set_title(f"GT B (nu={low_resol_coef})", fontsize=10)

    im_generated = axes[0, 2].imshow(gen_after, origin="upper", aspect="auto", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 2].set_title("Generated B (final)", fontsize=10)

    im_init = axes[0, 3].imshow(dataB_predict_initial, origin="upper", aspect="auto", cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 3].set_title("Initial prediction", fontsize=10)

    im_err = axes[0, 4].imshow(err_after, origin="upper", aspect="auto", cmap=cmap_err, vmin=vmin_e, vmax=vmax_e, interpolation="nearest")
    axes[0, 4].set_title("Masked error |GTB-Gen|", fontsize=10)
    axes[0, 4].set_facecolor("white")

    # Row 1: PDE maps (있으면), 없으면 빈칸 처리
    def _imshow_or_blank(ax, arr, title, vmin, vmax):
        if arr is None:
            ax.axis("off")
            ax.set_title(title + " (N/A)", fontsize=10)
            return None
        im = ax.imshow(arr, origin="upper", aspect="auto", cmap=cmap_err, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_facecolor("white")
        return im

    im_pde_gta = _imshow_or_blank(axes[1, 0], pde_map_gta, "GT A PDE residual map", vmin_gta, vmax_gta)
    im_pde_gtb = _imshow_or_blank(axes[1, 1], pde_map_gtb, "GT B PDE residual map", vmin_gtb, vmax_gtb)
    im_pde_pred = _imshow_or_blank(axes[1, 2], pde_map_pred, "Generated PDE residual map", vmin_pred, vmax_pred)
    im_pde_init = _imshow_or_blank(axes[1, 3], pde_map_init, "Initial PDE residual map", vmin_init, vmax_init)

    # PDE diff (GTB vs Gen) — 둘 다 있을 때만
    if (pde_map_pred is not None) and (pde_map_gtb is not None):
        pde_diff = np.abs(pde_map_pred - pde_map_gtb)
        if m0 is not None:
            pde_diff = pde_diff.copy()
            pde_diff[~m0] = np.nan
        im_pde_diff = axes[1, 4].imshow(pde_diff, origin="upper", aspect="auto", cmap=cmap_err,
                                        vmin=vmin_pred, vmax=vmax_pred, interpolation="nearest")
        axes[1, 4].set_title("PDE |Gen - GTB|", fontsize=10)
        axes[1, 4].set_facecolor("white")
    else:
        axes[1, 4].axis("off")
        axes[1, 4].set_title("PDE |Gen - GTB| (N/A)", fontsize=10)
        im_pde_diff = None

    # colorbars: 각 패널 옆에 붙이기 (너 원래 방식 유지)
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
