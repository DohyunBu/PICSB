from re import S
import torch
from torch import nn
import math as mt
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import t


def mse_loss(pred, target):
    mse = nn.MSELoss(reduction='none')
    return mse(pred, target)


def kinetic_energy(x):
    u_prime = ((x[..., 0] - x[..., 0].mean(dim=1, keepdim=True)) ** 2).mean(1)
    v_prime = ((x[..., 1] - x[..., 1].mean(dim=1, keepdim=True)) ** 2).mean(1)
    return 0.5 * (u_prime + v_prime)


def _compute_stats(x, confidence=0.95):
    x = x.detach().float().reshape(-1)

    x = x[torch.isfinite(x)]
    n = x.numel()

    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci": [float("nan"), float("nan")]
        }

    mean = x.mean()

    if n < 2:
        return {
            "mean": mean.item(),
            "std": 0.0,
            "ci": [mean.item(), mean.item()]
        }

    std = x.std(unbiased=True)
    se = std / math.sqrt(n)

    alpha = 1.0 - confidence
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * se.item()

    lower = mean.item() - margin
    upper = mean.item() + margin

    return {
        "mean": mean.item(),
        "std": std.item(),
        "ci": [lower, upper]
    }


def _spectral_error_per_sample(pred, target):
    b, t, h, w, c = pred.shape
    device = pred.device

    if t > 1 and h > 1 and w > 1:
        pred_F = torch.fft.fftn(pred, dim=[1, 2, 3])
        target_F = torch.fft.fftn(target, dim=[1, 2, 3])

        err_spec = torch.abs(pred_F - target_F) ** 2
        norm_spec = torch.abs(target_F) ** 2

        nbin = min(t // 2, h // 2, w // 2)
        if nbin < 1:
            nan_vec = torch.full((b,), float("nan"), device=device)
            return nan_vec, nan_vec, nan_vec, nan_vec, nan_vec, nan_vec, nan_vec

        err_F = torch.zeros([b, nbin, c], device=device)
        norm_F = torch.zeros([b, nbin, c], device=device)
        count_F = torch.zeros([nbin], device=device)

        for i in range(t // 2):
            for j in range(h // 2):
                for k in range(w // 2):
                    it = mt.floor(mt.sqrt(i**2 + j**2 + k**2))
                    if it > nbin - 1:
                        continue
                    err_F[:, it] += err_spec[:, i, j, k]
                    norm_F[:, it] += norm_spec[:, i, j, k]
                    count_F[it] += 1

        count_F = count_F.clamp_min(1.0).view(1, nbin, 1)
        err_F = torch.sqrt((err_F / count_F).mean(dim=2)) / (t * h * w)
        norm_F = torch.sqrt((norm_F / count_F).mean(dim=2)) / (t * h * w)

    elif t == 1 and h > 1 and w > 1:
        pred2 = pred[:, 0]   
        target2 = target[:, 0] 

        pred_F = torch.fft.fftn(pred2, dim=[1, 2])
        target_F = torch.fft.fftn(target2, dim=[1, 2])

        err_spec = torch.abs(pred_F - target_F) ** 2
        norm_spec = torch.abs(target_F) ** 2

        nbin = min(h // 2, w // 2)
        if nbin < 1:
            nan_vec = torch.full((b,), float("nan"), device=device)
            return nan_vec, nan_vec, nan_vec, nan_vec, nan_vec, nan_vec, nan_vec

        err_F = torch.zeros([b, nbin, c], device=device)
        norm_F = torch.zeros([b, nbin, c], device=device)
        count_F = torch.zeros([nbin], device=device)

        for j in range(h // 2):
            for k in range(w // 2):
                it = mt.floor(mt.sqrt(j**2 + k**2))
                if it > nbin - 1:
                    continue
                err_F[:, it] += err_spec[:, j, k]
                norm_F[:, it] += norm_spec[:, j, k]
                count_F[it] += 1

        count_F = count_F.clamp_min(1.0).view(1, nbin, 1)
        err_F = torch.sqrt((err_F / count_F).mean(dim=2)) / (h * w)
        norm_F = torch.sqrt((norm_F / count_F).mean(dim=2)) / (h * w)

    elif h == 1 and t > 1 and w > 1:
        pred2 = pred[:, :, 0]     
        target2 = target[:, :, 0] 

        pred_F = torch.fft.fftn(pred2, dim=[1, 2])
        target_F = torch.fft.fftn(target2, dim=[1, 2])

        err_spec = torch.abs(pred_F - target_F) ** 2
        norm_spec = torch.abs(target_F) ** 2

        nbin = min(t // 2, w // 2)
        if nbin < 1:
            nan_vec = torch.full((b,), float("nan"), device=device)
            return nan_vec, nan_vec, nan_vec, nan_vec, nan_vec, nan_vec, nan_vec

        err_F = torch.zeros([b, nbin, c], device=device)
        norm_F = torch.zeros([b, nbin, c], device=device)
        count_F = torch.zeros([nbin], device=device)

        for i in range(t // 2):
            for k in range(w // 2):
                it = mt.floor(mt.sqrt(i**2 + k**2))
                if it > nbin - 1:
                    continue
                err_F[:, it] += err_spec[:, i, k]
                norm_F[:, it] += norm_spec[:, i, k]
                count_F[it] += 1

        count_F = count_F.clamp_min(1.0).view(1, nbin, 1)
        err_F = torch.sqrt((err_F / count_F).mean(dim=2)) / (t * w)
        norm_F = torch.sqrt((norm_F / count_F).mean(dim=2)) / (t * w)

    else:
        nan_vec = torch.full((b,), float("nan"), device=device)
        return nan_vec, nan_vec, nan_vec, nan_vec, nan_vec, nan_vec, nan_vec

    iLow = int(np.round(nbin / 3))
    iHigh = int(np.round(nbin * 2 / 3))

    f_each = err_F.mean(dim=1)

    if iLow > 0:
        low_each = err_F[:, :iLow].mean(dim=1)
        rel_low_each = (
            err_F[:, :iLow] / norm_F[:, :iLow].clamp_min(1e-12)
        ).mean(dim=1)
    else:
        low_each = torch.zeros(b, device=device)
        rel_low_each = torch.zeros(b, device=device)

    if iHigh > iLow:
        mid_each = err_F[:, iLow:iHigh].mean(dim=1)
        rel_mid_each = (
            err_F[:, iLow:iHigh] / norm_F[:, iLow:iHigh].clamp_min(1e-12)
        ).mean(dim=1)
    else:
        mid_each = torch.zeros(b, device=device)
        rel_mid_each = torch.zeros(b, device=device)

    if iHigh < nbin:
        high_each = err_F[:, iHigh:].mean(dim=1)
        rel_high_each = (
            err_F[:, iHigh:] / norm_F[:, iHigh:].clamp_min(1e-12)
        ).mean(dim=1)
    else:
        high_each = torch.zeros(b, device=device)
        rel_high_each = torch.zeros(b, device=device)

    return (
        f_each, low_each, mid_each, high_each,
        rel_low_each, rel_mid_each, rel_high_each
    )


def eval_metrics(pred, target, c, batch_size=None):
    pred_all, target_all = pred[..., :c], target[..., :c]
    B, t, h, w, c = target_all.size()
    device = target.device

    if batch_size is None:
        batch_size = B

    rmse_list = []
    mae_list = []
    rel_l2_error_list = []
    r2_list = []
    ke_error_list = []
    f_error_list = []
    low_f_error_list = []
    mid_f_error_list = []
    high_f_error_list = []
    rel_low_f_error_list = []
    rel_mid_f_error_list = []
    rel_high_f_error_list = []
    freq_error_list = []

    for start in range(0, B, batch_size):
        pred = pred_all[start:start+batch_size]
        target = target_all[start:start+batch_size]
        b = pred.shape[0]

        diff = pred - target

        rmse_each = torch.sqrt((diff ** 2).reshape(b, -1).mean(dim=1))
        mae_each = diff.abs().reshape(b, -1).mean(dim=1)
        err_l2 = torch.norm(diff.reshape(b, -1), dim=1)
        norm = torch.norm(target.reshape(b, -1), dim=1).clamp_min(1e-12)
        rel_l2_each = err_l2 / norm
        target_flat = target.reshape(b, -1)
        pred_flat = pred.reshape(b, -1)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        ss_res = ((pred_flat - target_flat) ** 2).sum(dim=1)
        ss_tot = ((target_flat - target_mean) ** 2).sum(dim=1).clamp_min(1e-12)
        r2_each = 1.0 - ss_res / ss_tot

        if c < 2:
            ke_each = torch.zeros(b, device=device)
        else:
            pred_ke = kinetic_energy(pred)
            target_ke = kinetic_energy(target)
            ke_each = (pred_ke - target_ke).abs().reshape(b, -1).mean(dim=1)

        (f_each, low_each, mid_each, high_each,
            rel_low_each, rel_mid_each, rel_high_each) = _spectral_error_per_sample(pred, target)

        sum_pred = torch.sum(pred, dim=[2, 3, 4])  
        sum_target = torch.sum(target, dim=[2, 3, 4]) 
        sum_pred_F = torch.fft.fftn(sum_pred, dim=1)
        sum_target_F = torch.fft.fftn(sum_target, dim=1)
        freq_each = torch.abs(sum_pred_F - sum_target_F).mean(dim=1)

        rmse_list.append(rmse_each.detach())
        mae_list.append(mae_each.detach())
        rel_l2_error_list.append(rel_l2_each.detach())
        r2_list.append(r2_each.detach())
        ke_error_list.append(ke_each.detach())
        f_error_list.append(f_each.detach())
        low_f_error_list.append(low_each.detach())
        mid_f_error_list.append(mid_each.detach())
        high_f_error_list.append(high_each.detach())
        rel_low_f_error_list.append(rel_low_each.detach())
        rel_mid_f_error_list.append(rel_mid_each.detach())
        rel_high_f_error_list.append(rel_high_each.detach())
        freq_error_list.append(freq_each.detach())

    rmse_all = torch.cat(rmse_list, dim=0)
    mae_all = torch.cat(mae_list, dim=0)
    rel_l2_all = torch.cat(rel_l2_error_list, dim=0)
    r2_all = torch.cat(r2_list, dim=0)
    ke_all = torch.cat(ke_error_list, dim=0)
    f_all = torch.cat(f_error_list, dim=0)
    low_f_all = torch.cat(low_f_error_list, dim=0)
    mid_f_all = torch.cat(mid_f_error_list, dim=0)
    high_f_all = torch.cat(high_f_error_list, dim=0)
    rel_low_f_all = torch.cat(rel_low_f_error_list, dim=0)
    rel_mid_f_all = torch.cat(rel_mid_f_error_list, dim=0)
    rel_high_f_all = torch.cat(rel_high_f_error_list, dim=0)
    freq_all = torch.cat(freq_error_list, dim=0)

    results = {
        "RMSE": _compute_stats(rmse_all),
        "MAE": _compute_stats(mae_all),
        "Rel_L2": _compute_stats(rel_l2_all),
        "R2": _compute_stats(r2_all),
        "KE": _compute_stats(ke_all),
        "f_error": _compute_stats(f_all),
        "low_f_error": _compute_stats(low_f_all),
        "mid_f_error": _compute_stats(mid_f_all),
        "high_f_error": _compute_stats(high_f_all),
        "rel_low_f_error": _compute_stats(rel_low_f_all),
        "rel_mid_f_error": _compute_stats(rel_mid_f_all),
        "rel_high_f_error": _compute_stats(rel_high_f_all),
        "freq_error": _compute_stats(freq_all),
    }

    return results