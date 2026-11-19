import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from models.model import PICSB
from models.preprocessing import dataset_interpolation
from models.visualize import save_field_snapshot
from models.data_generation import generate_dataset, make_sensor_mask
from models.metrics import eval_metrics
from datasets.dataset import PDEPairDataset
from torch.utils.data import DataLoader
from utils import (
    load_yaml,
    merge_config,
    apply_defaults,
    set_seed,
    resolve_paths,
    build_masks,
)

def parse_args():
    p = argparse.ArgumentParser(description="PICSB inference")
    p.add_argument("--config_path", "--config", dest="config_path", type=str, default=None)
    p.add_argument("--dataset", choices=["burgers", "darcy", "kolmo"], default=None)
    p.add_argument("--regime", choices=["sparse", "sensor_random", "sensor_fix"], default=None, help="R1, R2, R3")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="Result/infer")
    p.add_argument("--viz_dir", type=str, default=None, help="Directory to store visualization SVGs")
    p.add_argument("--high_path", type=str, default=None)
    p.add_argument("--low_path", type=str, default=None)
    p.add_argument("--dataset_name", choices=["reaction_diffusion", "gray"], default=None)
    p.add_argument("--mask_path", type=str, default=None)
    p.add_argument("--force_cpu", action="store_true", default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument("--lf-grid", type=int, choices=[16, 32], default=None)
    return p.parse_args()

def _to_metric_tensor(x, dataset):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x = x.float()
    if dataset == "gray":
        return x.permute(0, 2, 3, 4, 1).contiguous(), x.shape[1]
    if dataset == "reaction_diffusion":
        return x[:, 0].unsqueeze(2).unsqueeze(-1).contiguous(), 1
    if dataset == "darcy":
        return x.unsqueeze(-1).contiguous(), x.shape[1]
    if dataset == "kolmo":
        return x.unsqueeze(-1).contiguous(), 1
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0].unsqueeze(2).unsqueeze(-1).contiguous(), 1
    if x.ndim == 4:
        return x.unsqueeze(-1).contiguous(), 1
    raise ValueError(f"Unsupported metric tensor shape: {tuple(x.shape)}")

def main():
    cli = parse_args()
    if getattr(cli, "dataset_name", None) is not None:
        cli.dataset = cli.dataset_name
    cfg = merge_config(cli, load_yaml(cli.config_path))
    cfg = apply_defaults(cfg)
    if cfg["dataset"] == "reaction_diffusion":
        cfg["high_path"] = cfg.get("high_path") or "./benchmarks/RD/ReacDiff_Nu5.0_Rho5.0.hdf5"
        cfg["low_path"] = cfg.get("low_path") or "./benchmarks/RD/low_f_1d.npy"
        cfg["model_path"] = cfg.get("model_path") or "./Result/model/RD_100_10_1e7.pt"
    if cfg["dataset"] == "gray":
        cfg["high_path"] = cfg.get("high_path") or "./benchmarks/gray/gray_high_f_1d_down.npy"
        cfg["low_path"] = cfg.get("low_path") or f"./benchmarks/gray/gray_low_f_1d_down_{cfg['regime']}.npy"
        cfg["mask_path"] = cfg.get("mask_path") or f"./benchmarks/gray/gray_mask_high_1d_{cfg['regime']}.npy"
        cfg["model_path"] = cfg.get("model_path") or "./Result/model/gray_100_10_1e2_sparse.pt"
    if cfg.get("model_path") is None:
        cfg["model_path"] = str(Path(cfg["save_dir"]) / cfg["model_name"])
    if cfg["dataset"] == "kolmo" and cli.model_path is None:
        model_path = Path(cfg["model_path"])
        fold_suffix = f"_fold{int(cfg.get('fold_idx', 0))}"
        if not model_path.stem.endswith(fold_suffix):
            model_path = model_path.with_name(f"{model_path.stem}{fold_suffix}{model_path.suffix}")
        cfg["model_path"] = str(model_path)
    if cfg.get("force_cpu"):
        torch.cuda.is_available = lambda: False
    set_seed(cfg["seed"])
    if cfg["dataset"] == "darcy":
        a, p = resolve_paths(cfg)
        mask_high_p = build_masks(cfg, p)

        low_p = dataset_interpolation(
            p,
            mask_high_p,
            interpol_target="darcy",
            seed=cfg["seed"],
        )
        high = np.concatenate([a, p], axis=1)
        low = np.concatenate([a, low_p], axis=1)
        mask_high = np.concatenate([mask_high_p, mask_high_p], axis=1)

        N_train = 32
        N_skip = 8
        low_test = low[N_train + N_skip:]
        high_test = high[N_train + N_skip:]
        mask_test = mask_high[N_train + N_skip:]

    elif cfg["dataset"] == "reaction_diffusion":
        import h5py
        high_path = cfg.get("high_path") or "./benchmarks/RD/ReacDiff_Nu5.0_Rho5.0.hdf5"
        with h5py.File(high_path, "r") as f:
            high = f["tensor"][:80, 36:100, :]
            high = high[:, None, :, ::16]
        mask_high = make_sensor_mask(high, n_sensors=None, sparse_ratio=0.1, seed=cfg["seed"])
        low = dataset_interpolation(high, mask_high, interpol_target="RD", seed=cfg["seed"])
        N_train = 32
        N_skip = 8
        low_test = low[N_train + N_skip:]
        high_test = high[N_train + N_skip:]
        mask_test = mask_high[N_train + N_skip:]

    elif cfg["dataset"] == "gray":
        high, low = resolve_paths(cfg)
        mask_high = np.load(cfg.get("mask_path") or "./benchmarks/gray/gray_mask_high_1d_sparse.npy", mmap_mode="r").copy()
        mask_high[:, :, :, 0, :] = True
        mask_high[:, :, :, -1, :] = True
        mask_high[:, :, :, :, 0] = True
        mask_high[:, :, :, :, -1] = True
        N_train = 32
        N_skip = 8
        low_test = low[N_train + N_skip:]
        high_test = high[N_train + N_skip:]
        mask_test = mask_high[N_train + N_skip:]

    elif cfg["dataset"] == "kolmo":
        if getattr(cli, "low_path", None) is None:
            cfg["low_path"] = "./data/kolmo/low_fidelity_10%_sparse.npy"
        high, low = resolve_paths(cfg)
        mask_high = make_sensor_mask(high, n_sensors=None, sparse_ratio=0.1, seed=cfg["seed"])
        test_folds = [
            np.arange(32, 40),
            np.arange(0, 8),
            np.arange(8, 16),
            np.arange(16, 24),
            np.arange(24, 32),
        ]
        test_idx = test_folds[int(cfg.get("fold_idx", 0))]
        low_test = low[test_idx]
        high_test = high[test_idx]
        mask_test = mask_high[test_idx]

    elif cfg["dataset"] == "burgers":
        lf_grid = cfg.get("lf_grid", 16)
        if lf_grid not in (16, 32):
            raise ValueError("lf_grid must be 16 or 32.")
        stride = 128 // lf_grid
        high = generate_dataset(n_samples=40, nu=0.01, Nx=128, Nt=128, T=1.0, L=1.0, seed=0, warmup_steps=100)
        low = generate_dataset(n_samples=40, nu=0.1, Nx=lf_grid, Nt=lf_grid, T=1.0, L=1.0, U0=high[:,:,::stride,::stride][:,:,[0], :], seed=20)
        low = F.interpolate(torch.as_tensor(low, dtype=torch.float32), size=(128, 128), mode="nearest").cpu().numpy()
        mask_high = make_sensor_mask(high, n_sensors=12, sparse_ratio=None, seed=0, sensor_random=False)

        high_extra = generate_dataset(n_samples=36, nu=0.01, Nx=128, Nt=128, T=1.0, L=1.0, seed=1999, warmup_steps=100)
        low_extra = generate_dataset(n_samples=36, nu=0.1, Nx=lf_grid, Nt=lf_grid, T=1.0, L=1.0, U0=high_extra[:,:,::stride,::stride][:,:,[0], :], seed=2026)
        low_extra = F.interpolate(torch.as_tensor(low_extra, dtype=torch.float32), size=(128, 128), mode="nearest").cpu().numpy()
        mask_extra = make_sensor_mask(high_extra, n_sensors=12, sparse_ratio=None, seed=0, sensor_random=False)
        low_test = np.concatenate([low[36:], low_extra], axis=0)
        high_test = np.concatenate([high[36:], high_extra], axis=0)
        mask_test = np.concatenate([mask_high[36:], mask_extra], axis=0)

    if cfg.get("max_test_samples") is not None:
        n_test = min(int(cfg["max_test_samples"]), low_test.shape[0])
        low_test = low_test[:n_test]
        high_test = high_test[:n_test]
        mask_test = mask_test[:n_test]
    ds_test = PDEPairDataset(low_data=low_test, high_data=high_test, mask_high_data=mask_test)
    dl_test = DataLoader(ds_test, batch_size=cfg["test_batch_size"], shuffle=False, drop_last=False)
    model = PICSB(
        train_dataloader=dl_test,
        timepoints=cfg["timepoints"],
        time_dim=cfg["time_dim"],
        steps=cfg["steps"],
        base_lr=cfg["base_lr"],
        beta_min=cfg["beta_min"],
        beta_max=cfg["beta_max"],
        pde_type=cfg["dataset"],
        in_channels=cfg["in_channels"],
    )
    state = torch.load(cfg["model_path"], map_location="cpu", weights_only=False)
    model.v_fore.load_state_dict(state["v_fore"])
    model.time_embed.load_state_dict(state["time_embed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.v_fore.to(device)
    model.time_embed.to(device)

    preds = []
    with torch.no_grad():
        for x0, x1, m1 in dl_test:
            x0 = x0.to(device)
            x1 = x1.to(device)
            m1 = m1.to(device) if m1 is not None else None
            if cfg["dataset"] == "darcy":
                x0 = x0[:, 1:2]
                x1 = x1[:, 1:2]
                m1 = m1[:, 1:2] if m1 is not None else None
            pred = model.eval_fore(x_0=x0, x_1=x1, mask_high=m1)
            preds.append(pred.cpu())

            if cfg.get("viz_dir"):
                viz_dir = Path(cfg["viz_dir"])
                print(viz_dir)
                for b in range(pred.shape[0]):
                    base = f"{cfg['dataset']}_test_{len(preds)-1:03d}_{b:02d}"
                    if cfg["dataset"] == "kolmo":
                        base = f"{cfg['dataset']}_fold{int(cfg.get('fold_idx', 0))}_test_{len(preds)-1:03d}_{b:02d}"
                    if cfg["dataset"] == "burgers":
                        save_field_snapshot(
                            x0[b:b+1].cpu(), batch_idx=0, vmin=-0.85, vmax=0.45,
                            out_svg=viz_dir / f"{base}_lf.svg", border_lw=3.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            x1[b:b+1].cpu(), batch_idx=0, vmin=-0.85, vmax=0.45,
                            out_svg=viz_dir / f"{base}_hf.svg", border_lw=3.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            pred[b:b+1].cpu(), batch_idx=0, vmin=-0.85, vmax=0.45,
                            out_svg=viz_dir / f"{base}_predict.svg", border_lw=3.0, cmap="coolwarm"
                        )
                    elif cfg["dataset"] == "darcy":
                        save_field_snapshot(
                            x0[b:b+1].cpu(), batch_idx=0, vmin=0.0, vmax=0.075,
                            out_svg=viz_dir / f"{base}_lf.svg", border_lw=3.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            x1[b:b+1].cpu(), batch_idx=0, vmin=0.0, vmax=0.075,
                            out_svg=viz_dir / f"{base}_hf.svg", border_lw=3.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            pred[b:b+1].cpu(), batch_idx=0, vmin=0.0, vmax=0.075,
                            out_svg=viz_dir / f"{base}_predict.svg", border_lw=3.0, cmap="coolwarm"
                        )
                    elif cfg["dataset"] == "kolmo":
                        snap = min(20, pred.shape[1]-1) if pred.ndim == 4 else 0
                        save_field_snapshot(
                            x0[b:b+1, snap:snap+1].cpu(), batch_idx=0, vmin=-16, vmax=16,
                            out_svg=viz_dir / f"{base}_lf.svg", border_lw=4.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            x1[b:b+1, snap:snap+1].cpu(), batch_idx=0, vmin=-16, vmax=16,
                            out_svg=viz_dir / f"{base}_hf.svg", border_lw=4.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            pred[b:b+1, snap:snap+1].cpu(), batch_idx=0, vmin=-16, vmax=16,
                            out_svg=viz_dir / f"{base}_predict.svg", border_lw=4.0, cmap="coolwarm"
                        )
                    elif cfg["dataset"] == "reaction_diffusion":
                        save_field_snapshot(
                            x0[b:b+1].cpu(), batch_idx=0, vmin=None, vmax=None,
                            out_svg=viz_dir / f"{base}_lf.svg", border_lw=3.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            x1[b:b+1].cpu(), batch_idx=0, vmin=None, vmax=None,
                            out_svg=viz_dir / f"{base}_hf.svg", border_lw=3.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            pred[b:b+1].cpu(), batch_idx=0, vmin=None, vmax=None,
                            out_svg=viz_dir / f"{base}_predict.svg", border_lw=3.0, cmap="coolwarm"
                        )
                    elif cfg["dataset"] == "gray":
                        snap = min(20, pred.shape[2]-1) if pred.ndim == 5 else 0
                        save_field_snapshot(
                            x0[b:b+1, 0:1, snap].cpu(), batch_idx=0, vmin=None, vmax=None,
                            out_svg=viz_dir / f"{base}_lf.svg", border_lw=3.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            x1[b:b+1, 0:1, snap].cpu(), batch_idx=0, vmin=None, vmax=None,
                            out_svg=viz_dir / f"{base}_hf.svg", border_lw=3.0, cmap="coolwarm"
                        )
                        save_field_snapshot(
                            pred[b:b+1, 0:1, snap].cpu(), batch_idx=0, vmin=None, vmax=None,
                            out_svg=viz_dir / f"{base}_predict.svg", border_lw=3.0, cmap="coolwarm"
                        )

    pred_all = torch.cat(preds, dim=0).numpy()

    gt = torch.tensor(high_test)
    if cfg["dataset"] == "darcy":
        gt = gt[:, 1:2] 
    pred_t = torch.tensor(pred_all)

    pred_metric, metric_c = _to_metric_tensor(pred_t, cfg["dataset"])
    gt_metric, _ = _to_metric_tensor(gt, cfg["dataset"])
    metric_results = eval_metrics(
        pred_metric,
        gt_metric,
        c=metric_c,
        batch_size=cfg.get("test_batch_size"),
    )
    print(f"RMSE: {metric_results['RMSE']['mean']:.4e}")
    print(f"MAE: {metric_results['MAE']['mean']:.4e}")
    print(f"Relative error: {metric_results['Rel_L2']['mean']:.4e}")
    print(f"fRMSE: {metric_results['f_error']['mean']:.4e}")

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    result_name = cfg["dataset"]
    if cfg["dataset"] == "kolmo":
        result_name = f"{result_name}_fold{int(cfg.get('fold_idx', 0))}"
    torch.save(pred_all, save_dir / f"{result_name}_pred.pt")
    torch.save(high_test, save_dir / f"{result_name}_high_gt.pt")
    torch.save(mask_test, save_dir / f"{result_name}_mask.pt")
    print("Inference done:", pred_all.shape)

if __name__ == "__main__":
    main()
