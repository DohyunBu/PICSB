import argparse
import os
from pathlib import Path
import numpy as np
from models.model import PICSB
from models.data_generation import make_sensor_mask
from datasets.dataset import PDEPairDataset
from torch.utils.data import DataLoader
from utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PICSB kolmo training.")
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
    parser.add_argument("--known-alpha", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--refresh", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--mask-keep-ratio", type=float, default=None)
    parser.add_argument("--mask-sparse-ratio", type=float, default=None)
    parser.add_argument("--mask-n-sensors", type=int, default=None)
    parser.add_argument("--mask-sensor-random", action="store_true", default=None)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    cfg = merge_config(cli_args, load_yaml(cli_args.config_path))
    cfg["dataset"] = "kolmo"
    if cli_args.regime is None:
        cfg["regime"] = "sparse"
    if cli_args.low_path is None:
        cfg["low_path"] = "./benchmarks/kolmo/low_fidelity_10%_sparse.npy"
    cfg = apply_defaults(cfg)
    cfg.setdefault("refresh", 100)
    set_seed(cfg["seed"])

    high, low = resolve_paths(cfg)
    mask_high = make_sensor_mask(high, n_sensors=None, sparse_ratio=0.1, seed=cfg["seed"])

    test_folds = [
        np.arange(32, 40),
        np.arange(0, 8),
        np.arange(8, 16),
        np.arange(16, 24),
        np.arange(24, 32),
    ]
    fold_idx = int(cfg.get("fold_idx", 0))
    test_idx = test_folds[fold_idx]
    all_idx = np.arange(low.shape[0])
    train_idx = np.setdiff1d(all_idx, test_idx)

    low_train = low[train_idx]
    high_train = high[train_idx]
    mask_train = mask_high[train_idx]

    low_test = low[test_idx]
    high_test = high[test_idx]
    mask_test = mask_high[test_idx]
    
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
    fold_suffix = f"_fold{fold_idx}"
    if not model_path.stem.endswith(fold_suffix):
        model_path = model_path.with_name(f"{model_path.stem}{fold_suffix}{model_path.suffix}")
    model.train(
        n_iterations=cfg["iterations"],
        model_name=str(model_path),
        known_alpha=cfg["known_alpha"],
        refresh=cfg["refresh"],
    )


if __name__ == "__main__":
    main()
