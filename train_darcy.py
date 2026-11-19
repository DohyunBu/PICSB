import argparse
import os
from pathlib import Path
import numpy as np
from models.model import PICSB
from models.preprocessing import dataset_interpolation
from datasets.dataset import PDEPairDataset
from torch.utils.data import DataLoader
from utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PICSB darcy training.")
    parser.add_argument("--config_path", type=str, help="YAML config path", default=None)
    parser.add_argument("--regime", choices=["sparse", "sensor_random", "sensor_fix"], default=None)
    parser.add_argument("--high_path", type=str, default=None)
    parser.add_argument("--low_path", type=str, default=None)
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
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--refresh", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--mask-keep-ratio", type=float, default=None)
    parser.add_argument("--mask-sparse-ratio", type=float, default=None)
    parser.add_argument("--mask-n-sensors", type=int, default=None)
    parser.add_argument("--mask-sensor-random", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    cfg = merge_config(cli_args, load_yaml(cli_args.config_path))
    cfg["dataset"] = "darcy"
    cfg = apply_defaults(cfg)
    cfg.setdefault("refresh", 100)
    set_seed(cfg["seed"])

    a, p = resolve_paths(cfg)

    mask_high = build_masks(cfg, p)

    low_p = dataset_interpolation(
        p,
        mask_high,
        interpol_target="darcy",
        noise_levels=None,
        seed=cfg["seed"],
    )
    high = np.concatenate([a, p], axis=1)
    low = np.concatenate([a, low_p], axis=1)
    mask_high = np.concatenate([mask_high, mask_high], axis=1)

    N_train = 32
    N_skip = 8

    low_train = low[:N_train]
    high_train = high[:N_train]
    mask_train = mask_high[:N_train]

    low_test = low[N_train + N_skip:]
    high_test = high[N_train + N_skip:]
    mask_test = mask_high[N_train + N_skip:]

    ds_train = PDEPairDataset(low_data=low_train, high_data=high_train, mask_high_data=mask_train)
    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)

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
