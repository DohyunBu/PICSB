from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

from models.data_generation import make_sensor_mask, make_sensor_mask_3d
from models.preprocessing import dataset_interpolation


def bicubic_fill_xy_gray(
    A: np.ndarray,
    mask: np.ndarray,
    *,
    min_points: int = 16,
    interpol_target: str = "gray",
    noise_levels=None,
    seed: int = 0,
) -> np.ndarray:
    if A.ndim != 5:
        raise ValueError(f"A must have shape (B, C, T, H, W), got {A.shape}")
    if mask.shape != A.shape:
        raise ValueError(f"mask shape must match A shape, got {mask.shape} vs {A.shape}")

    B, C, T, H, W = A.shape
    m = mask.astype(bool)
    A_filled = A.copy()

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pts_all = np.stack([yy.ravel(), xx.ravel()], axis=1)

    rng = np.random.default_rng(seed)

    for b in tqdm(range(B)):
        for c in range(C):
            for t in range(T):
                known = m[b, c, t]
                if known.all():
                    continue

                vals = A[b, c, t]
                n_known = int(known.sum())
                if n_known == 0:
                    continue

                pts_known = np.stack([yy[known], xx[known]], axis=1)
                v_known = vals[known]

                if noise_levels is not None:
                    sigma = np.std(v_known)
                    if sigma > 0:
                        v_known = v_known + rng.normal(
                            0.0, noise_levels * sigma, size=v_known.shape
                        )

                if interpol_target == "darcy":
                    filled = griddata(
                        pts_known, v_known, pts_all, method="nearest"
                    ).reshape(H, W)
                    vals_out = vals.copy()
                    vals_out[known] = v_known
                    vals_out[~known] = filled[~known]
                    A_filled[b, c, t] = vals_out
                    continue

                if n_known < min_points:
                    filled = griddata(
                        pts_known, v_known, pts_all, method="nearest"
                    ).reshape(H, W)
                else:
                    filled = griddata(
                        pts_known, v_known, pts_all, method="cubic"
                    ).reshape(H, W)
                    holes = (~known) & np.isnan(filled)
                    if holes.any():
                        filled_nearest = griddata(
                            pts_known, v_known, pts_all, method="nearest"
                        ).reshape(H, W)
                        filled[holes] = filled_nearest[holes]

                vals_out = vals.copy()
                vals_out[known] = v_known
                vals_out[~known] = filled[~known]
                A_filled[b, c, t] = vals_out

    return A_filled


def _default_output(dataset: str, regime: str, sparsity: float) -> str:
    if dataset == "kolmo":
        return f"./data/kolmo/low_fidelity_{int(sparsity * 100)}%_{regime}.npy"
    if dataset == "gray":
        return f"./data/gray/gray_low_f_1d_down_{regime}.npy"
    raise ValueError(f"Unsupported dataset {dataset}")


def _default_mask_output(dataset: str, regime: str) -> str | None:
    if dataset == "gray":
        return f"./data/gray/gray_mask_high_1d_{regime}.npy"
    return None


def _save(path: str, array: np.ndarray) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_obj, array)


def generate_low_fidelity(
    dataset: str,
    regime: str,
    sparsity: float,
    input_path: str,
    output_path: str | None = None,
    mask_output_path: str | None = None,
    seed: int = 0,
) -> tuple[str, str | None]:
    if not (0 < sparsity <= 1):
        raise ValueError("sparsity must be in the range (0, 1].")

    dataset = dataset.lower()
    regime = regime.lower()
    if regime not in {"sensor_random", "sensor_fix", "sparse"}:
        raise ValueError("regime must be one of: sensor_random, sensor_fix, sparse.")

    high = np.load(input_path, mmap_mode="r")

    if dataset == "kolmo":
        total_sensors = high.shape[-2] * high.shape[-1]
        if regime == "sensor_random":
            mask = make_sensor_mask(
                high,
                n_sensors=int(total_sensors * sparsity),
                sparse_ratio=None,
                seed=seed,
            )
        elif regime == "sensor_fix":
            mask = make_sensor_mask(
                high,
                n_sensors=int(total_sensors * sparsity),
                sparse_ratio=None,
                seed=seed,
                sensor_random=False,
            )
        else:
            mask = make_sensor_mask(
                high,
                n_sensors=None,
                sparse_ratio=sparsity,
                seed=seed,
            )
        low = dataset_interpolation(high, mask, interpol_target="kolmo", seed=seed)

    elif dataset == "gray":
        if high.ndim != 5:
            raise ValueError(f"Gray data must have shape (B, C, T, H, W), got {high.shape}")
        total_sensors = high.shape[-2] * high.shape[-1]
        if regime in {"sensor_random", "sensor_fix"}:
            mask = make_sensor_mask_3d(
                high,
                n_sensors=int(total_sensors * sparsity),
                mode=regime,
                sparse_ratio=None,
                seed=seed,
            )
        else:
            mask = make_sensor_mask_3d(
                high,
                n_sensors=None,
                mode="sparse",
                sparse_ratio=sparsity,
                seed=seed,
            )
        mask[:, :, :, 0, :] = True
        mask[:, :, :, -1, :] = True
        mask[:, :, :, :, 0] = True
        mask[:, :, :, :, -1] = True
        low = bicubic_fill_xy_gray(high, mask, seed=seed)

    else:
        raise ValueError("dataset must be one of: kolmo, gray.")

    if output_path is None:
        output_path = _default_output(dataset, regime, sparsity)
    if mask_output_path is None:
        mask_output_path = _default_mask_output(dataset, regime)

    _save(output_path, low)
    if mask_output_path is not None:
        _save(mask_output_path, mask)

    return output_path, mask_output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate low-fidelity Kolmogorov or Gray-Scott data."
    )
    parser.add_argument(
        "--dataset",
        choices=["kolmo", "gray"],
        required=True,
        help="Dataset to process.",
    )
    parser.add_argument(
        "--regime",
        choices=["sensor_random", "sensor_fix", "sparse"],
        default="sparse",
        help="Sampling regime to use.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.1,
        help="Fraction of sensors to keep.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the high-fidelity .npy file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional low-fidelity output path.",
    )
    parser.add_argument(
        "--mask-output",
        default=None,
        help="Optional mask output path. Defaults to a Gray-Scott mask path for gray.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed forwarded to mask generation.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    output, mask_output = generate_low_fidelity(
        dataset=args.dataset,
        regime=args.regime,
        sparsity=args.sparsity,
        input_path=args.input,
        output_path=args.output,
        mask_output_path=args.mask_output,
        seed=args.seed,
    )
    print(f"Saved low-fidelity dataset to: {output}")
    if mask_output is not None:
        print(f"Saved mask to: {mask_output}")


if __name__ == "__main__":
    main()
