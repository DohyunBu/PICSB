import numpy as np

def make_sensor_mask(
    data: np.ndarray,
    n_sensors: int = None,
    sparse_ratio: float = 0.1,
    seed: int = None,
) -> np.ndarray:
    assert data.ndim == 4
    B, C, H, W = data.shape
    rng = np.random.default_rng(seed)

    if C == 1:
        T, X = H, W
        mask = np.zeros((B, 1, T, X), dtype=bool)

        if n_sensors is not None:
            for b in range(B):
                x_idx = rng.choice(X, size=n_sensors, replace=False)
                mask[b, 0, :, x_idx] = True
        else:
            if sparse_ratio is None:
                raise ValueError("sparse_ratio cannot be None when n_sensors is None.")
            n = int(round(X * sparse_ratio))
            for b in range(B):
                for t in range(T):
                    x_idx = rng.choice(X, size=n, replace=False)
                    mask[b, 0, t, x_idx] = True

        return mask

    else:
        T, X, Y = C, H, W
        mask = np.zeros((B, T, X, Y), dtype=bool)

        if n_sensors is not None:
            for b in range(B):
                flat_idx = rng.choice(X * Y, size=n_sensors, replace=False)
                x_idx, y_idx = np.unravel_index(flat_idx, (X, Y))
                mask[b, :, x_idx, y_idx] = True
        else:
            if sparse_ratio is None:
                raise ValueError("sparse_ratio cannot be None when n_sensors is None.")
            n = int(round(X * Y * sparse_ratio))
            for b in range(B):
                for t in range(T):
                    flat_idx = rng.choice(X * Y, size=n, replace=False)
                    x_idx, y_idx = np.unravel_index(flat_idx, (X, Y))
                    mask[b, t, x_idx, y_idx] = True

        return mask