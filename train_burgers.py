import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from models.model import PICSB
from models.data_generation import generate_dataset, make_sensor_mask
from datasets.dataset import PDEPairDataset
from torch.utils.data import DataLoader
from utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PICSB burgers training.")
    parser.add_argument("--config_path", type=str, help="YAML config path", default=None)
    parser.add_argument("--lf-grid", type=int, choices=[16, 32], default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--timepoints", type=float, nargs="+", default=None)
    parser.add_argument("--time-dim", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--beta-min", type=float, default=None)
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--known-alpha", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--refresh", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    cfg = merge_config(cli_args, load_yaml(cli_args.config_path))
    cfg["dataset"] = "burgers"
    cfg["regime"] = "sensor_fix"
    cfg.setdefault("lf_grid", 16)
    if cfg["lf_grid"] not in (16, 32):
        raise ValueError("lf_grid must be 16 or 32.")
    if cfg.get("model_name") is None:
        cfg["model_name"] = f"burgers_{cfg['lf_grid']}.pt"
    cfg = apply_defaults(cfg)
    cfg.setdefault("refresh", 1)
    cfg.setdefault("shuffle", False)
    set_seed(cfg["seed"])

    high = generate_dataset(n_samples=40, nu=0.01, Nx=128, Nt=128, T=1.0, L=1.0, seed=0, warmup_steps=100)
    lf_grid = cfg["lf_grid"]
    stride = 128 // lf_grid
    low = generate_dataset(n_samples=40, nu=0.1, Nx=lf_grid, Nt=lf_grid, T=1.0, L=1.0, U0=high[:,:,::stride,::stride][:,:,[0], :], seed=20)
    low = F.interpolate(torch.as_tensor(low, dtype=torch.float32), size=(128, 128), mode="nearest").cpu().numpy()
    mask_high = make_sensor_mask(high, n_sensors=12, sparse_ratio=None, seed=0, sensor_random=False)

    N_train = int(low.shape[0] * cfg["train_ratio"])
    N_skip = int(low.shape[0] * cfg["test_skip_ratio"])

    low_train = low[:N_train]
    high_train = high[:N_train]
    mask_train = mask_high[:N_train]

    low_test = low[N_train + N_skip:]
    high_test = high[N_train + N_skip:]
    mask_test = mask_high[N_train + N_skip:]

    ds_train = PDEPairDataset(low_data=low_train, high_data=high_train, mask_high_data=mask_train)
    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=cfg["shuffle"], drop_last=False)

    ds_test = PDEPairDataset(low_data=low_test, high_data=high_test, mask_high_data=mask_test)
    dl_test = DataLoader(ds_test, batch_size=cfg["test_batch_size"], shuffle=False, drop_last=False)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    model = PICSB(
        train_dataloader=dl_train,
        timepoints=cfg["timepoints"],
        time_dim=cfg["time_dim"],
        steps=cfg["steps"],
        base_lr=cfg["base_lr"],
        beta_min=cfg["beta_min"],
        beta_max=cfg["beta_max"],
        pde_type=cfg["dataset"],
        in_channels=cfg["in_channels"],
    )

    model_path = Path(cfg["save_dir"]) / cfg["model_name"]
    model.train(
        n_iterations=cfg["iterations"],
        model_name=str(model_path),
        known_alpha=cfg["known_alpha"],
        refresh=cfg["refresh"],
    )


if __name__ == "__main__":
    main()
