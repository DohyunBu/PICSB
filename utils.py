import argparse
import random
from models.data_generation import (
    make_sensor_mask,
    select_finetune_sample,
    generate_dataset,
)
from models.data_generation import make_sensor_mask_3d
from pathlib import Path
import yaml
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from datasets.dataset import PDEPairDataset

DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "burgers": {
        "batch_size": 8,
        "test_batch_size": 4,
        "in_channels": 1,
        "base_lr": 1e-4,
        "beta_min": 1e-2,
        "beta_max": 1e-2,
        "known_alpha": 0.01,
        "mask_n_sensors": 12,
        "mask_sparse_ratio": 0.1,
    },
    "darcy": {
        "batch_size": 8,
        "test_batch_size": 4,
        "in_channels": 1,
        "base_lr": 1e-6,
        "beta_min": 1e-6,
        "beta_max": 1e-6,
        "known_alpha": None,
        "mask_n_sensors": 128 * 12,
        "mask_sparse_ratio": 0.1,
    },
    "kolmo": {
        "batch_size": 4,
        "test_batch_size": 4,
        "in_channels": 40,
        "base_lr": 1e-3,
        "beta_min": 1e-2,
        "beta_max": 1e-2,
        "known_alpha": 1000,
        "mask_n_sensors": 256 * 25,
        "mask_sparse_ratio": 0.1,
    },
    "reaction_diffusion": {
        "batch_size": 8,
        "test_batch_size": 4,
        "in_channels": 1,
        "base_lr": 1e-4,
        "beta_min": 1e-7,
        "beta_max": 1e-7,
        "known_alpha": {"nu": 5.0, "rho": 5.0},
        "mask_n_sensors": None,
        "mask_sparse_ratio": 0.1,
    },
    "gray_scott": {
        "batch_size": 8,
        "test_batch_size": 4,
        "in_channels": 2,
        "base_lr": 1e-4,
        "beta_min": 1e-2,
        "beta_max": 1e-2,
        "known_alpha": {
            "delta_a": 2.0e-5,
            "delta_b": 1.0e-5,
            "feed": 0.098,
            "kill": 0.057,
            "t1": 630.0,
        },
        "mask_n_sensors": int(64 * 64 * 0.1),
        "mask_sparse_ratio": 0.1,
    },
}
DATASET_DEFAULTS["gray"] = DATASET_DEFAULTS["gray_scott"]


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path_obj.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(cli: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**cfg}
    for k, v in vars(cli).items():
        if v is not None:
            merged[k] = v
    return merged


def apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "dataset" not in cfg or cfg["dataset"] is None:
        raise ValueError("dataset must be set via CLI or config.")
    ds = cfg["dataset"]

    base = DATASET_DEFAULTS[ds]
    defaults = {
        "regime": "sparse",
        "timepoints": [0.0, 1.0],
        "time_dim": 8,
        "steps": 10,
        "iterations": 100000,
        "save_dir": "Result/model",
        "model_name": f"{ds}.pt",
        "train_ratio": 0.8,
        "test_skip_ratio": 0.1,
        "num_workers": 0,
        "mask_keep_ratio": 1.0,
        "darcy_noise_levels": None,
        # burgers generation defaults
        "burgers_high_resol": 0.01,
        "burgers_low_resol": 0.1,
        "burgers_high_nx": 128,
        "burgers_high_nt": 128,
        "lf_grid": 16,
        "burgers_low_nx": 16,
        "burgers_low_nt": 16,
        "burgers_warmup": 100,
        "burgers_seed_high": 0,
        "burgers_seed_low": 0,
        "burgers_samples": 40,
    }

    for k, v in {**base, **defaults}.items():
        cfg.setdefault(k, v)

    if ds == "burgers":
        if cfg["lf_grid"] not in (16, 32):
            raise ValueError("lf_grid must be 16 or 32.")
        if cfg["burgers_high_nx"] != 128 or cfg["burgers_high_nt"] != 128:
            raise ValueError("Burgers HF grid must be 128 x 128.")
        cfg["burgers_low_nx"] = cfg["lf_grid"]
        cfg["burgers_low_nt"] = cfg["lf_grid"]

    if cfg.get("batch_size") is None:
        cfg["batch_size"] = base["batch_size"]
    if cfg.get("test_batch_size") is None:
        cfg["test_batch_size"] = base["test_batch_size"]
    if cfg.get("base_lr") is None:
        cfg["base_lr"] = base["base_lr"]
    if cfg.get("beta_min") is None:
        cfg["beta_min"] = base["beta_min"]
    if cfg.get("beta_max") is None:
        cfg["beta_max"] = base["beta_max"]
    if cfg.get("known_alpha") is None:
        cfg["known_alpha"] = base["known_alpha"]
    if cfg.get("in_channels") is None:
        cfg["in_channels"] = base["in_channels"]
    if cfg.get("mask_n_sensors") is None:
        cfg["mask_n_sensors"] = base["mask_n_sensors"]
    if cfg.get("mask_sparse_ratio") is None:
        cfg["mask_sparse_ratio"] = base["mask_sparse_ratio"]

    if cfg["dataset"] == "kolmo" and cfg.get("kolmo_low_percent") is None:
        cfg["kolmo_low_percent"] = 10.0

    if cfg.get("mask_keep_ratio") is None:
        cfg["mask_keep_ratio"] = 1.0

    if cfg.get("seed") is None:
        cfg["seed"] = 0

    if cfg.get("model_name") is None:
        cfg["model_name"] = f"{cfg['dataset']}_{cfg['regime']}.pt"

    if cfg.get("save_dir") is None:
        cfg["save_dir"] = "Result/model"

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_paths(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    ds = cfg["dataset"]

    if ds == "burgers":
        high = generate_dataset(
            n_samples=cfg["burgers_samples"],
            nu=cfg["burgers_high_resol"],
            Nx=cfg["burgers_high_nx"],
            Nt=cfg["burgers_high_nt"],
            T=1.0,
            L=1.0,
            seed=cfg["burgers_seed_high"],
            warmup_steps=cfg["burgers_warmup"],
        )
        stride = cfg["burgers_high_nx"] // cfg["lf_grid"]
        low = generate_dataset(
            n_samples=cfg["burgers_samples"],
            nu=cfg["burgers_low_resol"],
            Nx=cfg["burgers_low_nx"],
            Nt=cfg["burgers_low_nt"],
            T=1.0,
            L=1.0,
            seed=cfg["burgers_seed_low"],
            U0=high[:, :, [0], ::stride],
            warmup_steps=cfg["burgers_warmup"],
        )
        low = torch.nn.functional.interpolate(
            torch.as_tensor(low, dtype=torch.float32),
            size=(cfg["burgers_high_nt"], cfg["burgers_high_nx"]),
            mode="nearest",
        ).cpu().numpy()
        return high, low

    if ds == "darcy":
        a_path = cfg.get("a_path") or cfg.get("darcy_a_path") or "./data/darcy/lognorm_a_data.npy"
        p_path = cfg.get("p_path") or cfg.get("darcy_p_path") or "./data/darcy/lognorm_p_data.npy"
        if not Path(a_path).exists():
            a_path = "./benchmarks/darcy/lognorm_a_data.npy"
        if not Path(p_path).exists():
            p_path = "./benchmarks/darcy/lognorm_p_data.npy"
        a = np.load(a_path, mmap_mode="r")
        p = np.load(p_path, mmap_mode="r")
        
        return a, p

    if ds == "kolmo":
        high_path = cfg.get("high_path") or "./data/kolmo/kf_2d_re1000_256_40seed.npy"
        if not Path(high_path).exists():
            high_path = "./benchmarks/kolmo/kf_2d_re1000_256_40seed.npy"
        high = np.load(high_path, mmap_mode="r")
        high = high[:40,40:80,:,:]
        low_path = cfg["low_path"]
        if not Path(low_path).exists():
            low_path = str(Path("./benchmarks") / Path(low_path).relative_to("data"))
        low = np.load(low_path, mmap_mode="r")
        if low.shape == (40, 320, 256, 256):
            low = low[:40,40:80,:,:]
            
        return high, low

    if ds == "reaction_diffusion":
        import h5py
        high_path = cfg.get("high_path") or "./data/RD/ReacDiff_Nu5.0_Rho5.0.hdf5"
        low_path = cfg.get("low_path") or "./data/RD/low_f_1d.npy"
        with h5py.File(high_path, "r") as f:
            high = f["tensor"][:80, 36:100, :]
            high = high[:, None, :, ::16]
        low = np.load(low_path, mmap_mode="r")
        return high, low

    if ds == "gray":
        high_path = cfg.get("high_path") or "./data/gray/gray_high_f_1d_down.npy"
        low_path = cfg.get("low_path") or f"./data/gray/gray_low_f_1d_down_{cfg['regime']}.npy"
        high = np.load(high_path, mmap_mode="r")
        low = np.load(low_path, mmap_mode="r")
        return high, low

    raise ValueError(f"Unsupported dataset {ds}")

def build_masks(
    cfg: Dict[str, Any],
    high: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
        
    n_sensors = cfg["mask_n_sensors"]
    sparse_ratio = cfg["mask_sparse_ratio"] if cfg["regime"] == "sparse" else None
    sensor_random = (
        cfg["mask_sensor_random"] if cfg.get("mask_sensor_random") is not None else cfg["regime"] == "sensor_random"
    )
    mask_type = "darcy" if cfg["dataset"] == "darcy" else "else"

    if cfg["regime"] == "sparse":
        n_sensors = None
        if sparse_ratio is None:
            raise ValueError("sparse regime requires mask_sparse_ratio")

    if cfg["dataset"] == "gray":
        mask_path = cfg.get("mask_path") or f"./data/gray/gray_mask_high_1d_{cfg['regime']}.npy"
        if Path(mask_path).exists():
            mask_high = np.load(mask_path, mmap_mode="r").copy()
        else:
            mask_high = make_sensor_mask_3d(
                high,
                n_sensors=n_sensors,
                sparse_ratio=sparse_ratio,
                mode=cfg["regime"],
                seed=cfg["seed"],
            )
        mask_high[:, :, :, 0, :] = True
        mask_high[:, :, :, -1, :] = True
        mask_high[:, :, :, :, 0] = True
        mask_high[:, :, :, :, -1] = True
        return mask_high

    mask_high = make_sensor_mask(
        high,
        n_sensors=n_sensors,
        sparse_ratio=sparse_ratio,
        sensor_random=sensor_random,
        seed=cfg["seed"],
        type=mask_type,
    )

    keep = float(cfg.get("mask_keep_ratio", 1.0))
    if keep < 1.0:
        mask_high = select_finetune_sample(mask_high, keep_ratio=keep, seed=cfg["seed"], mode=cfg["regime"])

    if mask_type == 'darcy':
        mask_high[:, :, 0, :] = True
        mask_high[:, :, -1, :] = True
        mask_high[:, :, :, 0] = True
        mask_high[:, :, :, -1] = True
    return mask_high


def make_loaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    ds_train = PDEPairDataset(*splits["train"])
    ds_test = PDEPairDataset(*splits["test"])

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["batch_size"],
        shuffle=cfg.get("shuffle", True),
        drop_last=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=cfg["test_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    return dl_train, dl_test
