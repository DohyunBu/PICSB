import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch

from models.model import PICSB
from models.preprocessing import dataset_interpolation
from models.data_generation import select_finetune_sample
from utils import (
    load_yaml,
    merge_config,
    apply_defaults,
    set_seed,
    resolve_paths,
    build_masks,
    make_loaders,
)

FINETUNE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "burgers": dict(
        finetune_keep_ratio=0.1,
        finetune_lr=1e-6,
        finetune_iterations=20000,
        finetune_mode="sparse",
    ),
    "darcy": dict(
        finetune_keep_ratio=0.5,
        finetune_lr=1e-8,
        finetune_iterations=20000,
        finetune_mode="sparse",
    ),
    "kolmo": dict(
        finetune_keep_ratio=0.1,
        finetune_lr=1e-5,
        finetune_iterations=20000,
        finetune_mode="sparse",
        low_path=None,
    ),
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PICSB finetuning")
    p.add_argument("--config_path", "--config", dest="config_path", type=str, default=None)
    p.add_argument("--dataset", choices=["burgers", "darcy", "kolmo"], default=None)
    p.add_argument("--base_model_path", type=str, required=True, help="Pretrained checkpoint to start finetuning from")
    p.add_argument("--save_dir", type=str, default="Result/model_finetune")
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--regime", choices=["sparse", "sensor_random", "sensor_fix"], default=None, help="R1, R2, R3")
    p.add_argument("--finetune_keep_ratio", type=float, default=None)
    p.add_argument("--finetune_keep_ratios", type=float, nargs="+", default=None, help="Integrated finetune keep ratios (e.g., 1 0.5 0.1)")
    p.add_argument("--finetune_lr", type=float, default=None)
    p.add_argument("--finetune_iterations", type=int, default=None)
    p.add_argument("--finetune_mode", type=str, default=None)
    p.add_argument("--high_path", type=str, default=None)
    p.add_argument("--low_path", type=str, default=None)
    return p.parse_args()


def apply_finetune_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ds = cfg["dataset"]
    d = FINETUNE_DEFAULTS[ds]
    for k, v in d.items():
        cfg.setdefault(k, v)

    if cfg.get("finetune_keep_ratios") is None:
        cfg["finetune_keep_ratios"] = [1.0, 0.5, 0.1]
    if cfg.get("finetune_keep_ratio") is not None and cfg.get("finetune_keep_ratios") is None:
        cfg["finetune_keep_ratios"] = [cfg["finetune_keep_ratio"]]

    if cfg.get("finetune_mode") is None and cfg.get("regime") is not None:
        cfg["finetune_mode"] = cfg["regime"]

    if cfg.get("model_name") is None:
        base = Path(cfg["base_model_path"]).stem
        cfg["model_name"] = f"{base}_finetune.pt"
    return cfg

def build_finetune_masks(cfg: Dict[str, Any], high: np.ndarray) -> List[np.ndarray]:
    base_mask = build_masks(cfg, high)
    mode = cfg["finetune_mode"]
    masks = []
    for keep in cfg["finetune_keep_ratios"]:
        m = select_finetune_sample(base_mask, keep_ratio=keep, seed=cfg["seed"], mode=mode)
        if cfg["dataset"] == "darcy":
            m[:, :, 0, :] = True
            m[:, :, -1, :] = True
            m[:, :, :, 0] = True
            m[:, :, :, -1] = True
        masks.append(m)
    return masks

def split_by_level_and_concat(
    lows: List[np.ndarray],
    highs: List[np.ndarray],
    masks: List[np.ndarray],
    train_ratio: float,
    val_ratio: float,
):
    B = lows[0].shape[0]
    N_train = int(B * train_ratio)
    N_val = int(B * val_ratio)
    idx_train = slice(0, N_train)
    idx_val = slice(N_train, N_train + N_val)
    idx_test = slice(N_train + N_val, B)

    def cat_slice(idx):
        low_cat = np.concatenate([lv[idx] for lv in lows], axis=0)
        high_cat = np.concatenate([hv[idx] for hv in highs], axis=0)
        mask_cat = np.concatenate([mv[idx] for mv in masks], axis=0)
        return low_cat, high_cat, mask_cat

    low_train, high_train, mask_train = cat_slice(idx_train)
    low_val, high_val, mask_val = cat_slice(idx_val)
    low_test, high_test, mask_test = cat_slice(idx_test)
    return {
        "train": (low_train, high_train, mask_train),
        "val": (low_val, high_val, mask_val),
        "test": (low_test, high_test, mask_test),
    }

def prepare_finetune_data(cfg: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ds = cfg["dataset"]

    if ds == "burgers":
        high, low = resolve_paths(cfg)
        masks = build_finetune_masks(cfg, high)
        lows = [low for _ in masks]
        highs = [high for _ in masks]
        return split_by_level_and_concat(lows, highs, masks, cfg["train_ratio"], cfg.get("val_ratio", cfg["test_skip_ratio"]))

    if ds == "darcy":
        a, p = resolve_paths(cfg)
        masks_p = build_finetune_masks(cfg, p)

        lows = []
        highs = []
        masks_high = []
        for mask_p in masks_p:
            low_p = dataset_interpolation(
                p,
                mask_p,
                interpol_target="darcy",
                noise_levels=cfg.get("darcy_noise_levels"),
                seed=cfg["seed"],
            )
            lows.append(np.concatenate([a, low_p], axis=1))
            highs.append(np.concatenate([a, p], axis=1))
            masks_high.append(np.concatenate([mask_p, mask_p], axis=1))

        return split_by_level_and_concat(lows, highs, masks_high, cfg["train_ratio"], cfg.get("val_ratio", cfg["test_skip_ratio"]))

    if ds == "kolmo":
        high, low_base = resolve_paths(cfg)
        masks = build_finetune_masks(cfg, high)

        low_paths = cfg.get("low_path")
        if isinstance(low_paths, str):
            low_paths = [p.strip() for p in low_paths.split(",")]
        elif low_paths is None:
            low_paths = []
        if low_paths:
            loaded = []
            for p in low_paths:
                if Path(p).exists():
                    arr = np.load(p, mmap_mode="r")
                    if arr.shape[0] > high.shape[0]:
                        arr = arr[: high.shape[0]]
                    loaded.append(arr)
            if len(loaded) == 1 and len(masks) > 1:
                loaded = [loaded[0] for _ in masks]
            lows = [x[:40, 40:80, :, :] for x in loaded]
            highs = [high for _ in masks]
            masks_high = list(masks)
        
        else:
            lows = []
            highs = []
            masks_high = []
            for mask_ in masks:
                low_k = dataset_interpolation(
                    high,
                    mask_,
                    interpol_target="darcy",
                    noise_levels=cfg.get("darcy_noise_levels"),
                    seed=cfg["seed"],
                )
                lows.append(low_k)
                highs.append(high)
                masks_high.append(mask_)

        return split_by_level_and_concat(lows, highs, masks_high, cfg["train_ratio"], cfg.get("val_ratio", cfg["test_skip_ratio"]))

    raise ValueError(f"Unsupported dataset {ds}")


def load_pretrained(model: PICSB, path: str, lr: float) -> None:
    state = torch.load(path, map_location="cuda", weights_only=False)
    if "v_fore" in state:
        model.v_fore.load_state_dict(state["v_fore"])
    if "time_embed" in state:
        model.time_embed.load_state_dict(state["time_embed"])
    if hasattr(model, "optimizer_fore") and "optimizer_fore" in state:
        model.optimizer_fore.load_state_dict(state["optimizer_fore"])
    for pg in model.optimizer_fore.param_groups:
        pg["lr"] = lr


def main() -> None:
    cli = parse_args()
    cfg = merge_config(cli, load_yaml(cli.config_path))
    cfg = apply_defaults(cfg)
    cfg = apply_finetune_defaults(cfg)
    set_seed(cfg["seed"])

    splits = prepare_finetune_data(cfg)
    dl_train, _ = make_loaders(splits, cfg)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    model = PICSB(
        train_dataloader=dl_train,
        timepoints=cfg["timepoints"],
        time_dim=cfg["time_dim"],
        steps=cfg["steps"],
        base_lr=cfg["finetune_lr"],
        beta_min=cfg["beta_min"],
        beta_max=cfg["beta_max"],
        pde_type=cfg["dataset"],
        in_channels=cfg["in_channels"],
    )

    load_pretrained(model, cfg["base_model_path"], lr=cfg["finetune_lr"])

    model_path = Path(cfg["save_dir"]) / cfg["model_name"]
    model.train(
        n_iterations=cfg["finetune_iterations"],
        model_name=str(model_path),
        known_alpha=cfg["known_alpha"],
    )

if __name__ == "__main__":
    main()
