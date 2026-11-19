from __future__ import annotations
from tqdm import tqdm
import numpy as np

def make_sensor_mask_3d(
    data: np.ndarray,
    n_sensors: int | None = None,
    mode: str = "sensor_fix",
    sparse_ratio: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    if data.ndim != 5:
        raise ValueError(f"Expected data.ndim == 5, got {data.ndim}")

    B, C, T, H, W = data.shape
    rng = np.random.default_rng(seed)

    total_spatial = H * W
    mask = np.zeros((B, C, T, H, W), dtype=bool)

    if n_sensors is None:
        if sparse_ratio is None:
            raise ValueError("sparse_ratio cannot be None when n_sensors is None.")
        n_sensors = int(round(total_spatial * sparse_ratio))

    if n_sensors < 0 or n_sensors > total_spatial:
        raise ValueError(
            f"n_sensors must satisfy 0 <= n_sensors <= H*W ({total_spatial}), got {n_sensors}"
        )

    if mode == "sensor_fix":
        flat_idx = rng.choice(total_spatial, size=n_sensors, replace=False)
        h_idx, w_idx = np.unravel_index(flat_idx, (H, W))
        mask[:, :, :, h_idx, w_idx] = True

    elif mode == "sensor_random":
        for b in range(B):
            flat_idx = rng.choice(total_spatial, size=n_sensors, replace=False)
            h_idx, w_idx = np.unravel_index(flat_idx, (H, W))
            mask[b, :, :, h_idx, w_idx] = True

    elif mode == "sparse":
        for b in range(B):
            for t in range(T):
                flat_idx = rng.choice(total_spatial, size=n_sensors, replace=False)
                h_idx, w_idx = np.unravel_index(flat_idx, (H, W))
                mask[b, :, t, h_idx, w_idx] = True

    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            f"Supported modes are ['sensor_fix', 'sensor_random', 'sparse']"
        )

    return mask

def make_sensor_mask(
    data: np.ndarray,
    n_sensors: int = None,
    sensor_random: bool = True,
    sparse_ratio: float = 0.1,
    seed: int = None,
    type: str = 'else'
) -> np.ndarray:
    assert data.ndim == 4
    B, C, H, W = data.shape
    rng = np.random.default_rng(seed)

    if C == 1:
        T, X = H, W
        mask = np.zeros((B, 1, T, X), dtype=bool)

        if n_sensors is not None:
            if sensor_random:
                for b in range(B):
                    x_idx = rng.choice(X, size=n_sensors, replace=False)
                    mask[b, 0, :, x_idx] = True
            else:
                if type=='else':
                    x_idx = rng.choice(X, size=n_sensors, replace=False)
                    mask[:, 0, :, x_idx] = True
                elif type=='darcy':
                    flat_idx = rng.choice(T * X, size=n_sensors, replace=False)
                    x_idx, y_idx = np.unravel_index(flat_idx, (T, X))
                    mask[:, 0, x_idx, y_idx] = True
        else:
            if sparse_ratio is None:
                raise ValueError("sparse_ratio cannot be None when n_sensors is None.")
            if type=='else':
                n = int(round(X * sparse_ratio))
                for b in range(B):
                    for t in range(T):
                        x_idx = rng.choice(X, size=n, replace=False)
                        mask[b, 0, t, x_idx] = True
            elif type=='darcy':
                n = int(round(T * X * sparse_ratio))
                for b in range(B):
                    flat_idx = rng.choice(T * X, size=n, replace=False)
                    x_idx, y_idx = np.unravel_index(flat_idx, (T, X))
                    mask[b, 0, x_idx, y_idx] = True

        return mask

    else:
        T, X, Y = C, H, W
        mask = np.zeros((B, T, X, Y), dtype=bool)

        if n_sensors is not None:
            if sensor_random:
                for b in range(B):
                    flat_idx = rng.choice(X * Y, size=n_sensors, replace=False)
                    x_idx, y_idx = np.unravel_index(flat_idx, (X, Y))
                    mask[b, :, x_idx, y_idx] = True
            else:
                flat_idx = rng.choice(X * Y, size=n_sensors, replace=False)
                x_idx, y_idx = np.unravel_index(flat_idx, (X, Y))
                mask[:, :, x_idx, y_idx] = True
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

def select_finetune_sample(
    mask: np.ndarray,
    keep_ratio: float = 0.5,
    seed: int = None,
    mode: str = "sensor_fix"
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep_ratio = float(keep_ratio)

    B, C, H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    def _subsample_flat_indices(flat_idx: np.ndarray, n_keep: int):
        if flat_idx.size == 0 or n_keep <= 0:
            return np.empty((0,), dtype=np.int64)
        if n_keep >= flat_idx.size:
            return flat_idx
        sel = rng.choice(flat_idx.size, size=n_keep, replace=False)
        return flat_idx[sel]

    if C == 1:
        if mode == "sparse":
            for b in range(B):
                flat = np.flatnonzero(mask[b, 0])
                n_keep = int(round(flat.size * keep_ratio))
                chosen = _subsample_flat_indices(flat, n_keep)
                out_b = out[b, 0].reshape(-1)
                out_b[chosen] = True
            return out
        col_struct = False
        for b in range(B):
            if np.any(mask[b, 0].sum(axis=0) == H):
                col_struct = True
                break

        # ---------- Burgers: (B,1,T,X) ----------
        if col_struct:
            T, X = H, W

            if mode == "sensor_fix":
                x_candidates = np.where((mask[:, 0].sum(axis=1) == T).any(axis=0))[0]
                n_keep = int(round(x_candidates.size * keep_ratio))
                if n_keep <= 0:
                    return out
                choose = rng.choice(x_candidates, size=n_keep, replace=False)
                out[:, 0, :, choose] = True
                return out

            if mode == "sensor_random":
                for b in range(B):
                    x_candidates = np.where(mask[b, 0].sum(axis=0) == T)[0]
                    n_keep = int(round(x_candidates.size * keep_ratio))
                    if n_keep <= 0:
                        continue
                    choose = rng.choice(x_candidates, size=n_keep, replace=False)
                    out[b, 0, :, choose] = True
                return out

            for b in range(B):
                flat = np.flatnonzero(mask[b, 0])
                n_keep = int(round(flat.size * keep_ratio))
                chosen = _subsample_flat_indices(flat, n_keep)
                out_b = out[b, 0].reshape(-1)
                out_b[chosen] = True
            return out

        # ---------- Darcy: (B,1,X,Y) ----------
        X, Y = H, W

        if mode == "sensor_fix":
            xy_map = mask[:, 0].any(axis=0)  
            flat = np.flatnonzero(xy_map)
            n_keep = int(round(flat.size * keep_ratio))
            chosen = _subsample_flat_indices(flat, n_keep)
            xy_out = out[:, 0].reshape(B, -1)
            xy_out[:, chosen] = True
            return out

        if mode == "sensor_random":
            for b in range(B):
                flat = np.flatnonzero(mask[b, 0])
                n_keep = int(round(flat.size * keep_ratio))
                chosen = _subsample_flat_indices(flat, n_keep)
                out_b = out[b, 0].reshape(-1)
                out_b[chosen] = True
            return out

        for b in range(B):
            flat = np.flatnonzero(mask[b, 0])
            n_keep = int(round(flat.size * keep_ratio))
            chosen = _subsample_flat_indices(flat, n_keep)
            out_b = out[b, 0].reshape(-1)
            out_b[chosen] = True
        return out

    # ---------- Kolmo: (B,T,X,Y) (T=C, X=H, Y=W) ----------
    T, X, Y = C, H, W

    if mode == "sparse":
        for b in range(B):
            flat = np.flatnonzero(mask[b])
            n_keep = int(round(flat.size * keep_ratio))
            chosen = _subsample_flat_indices(flat, n_keep)
            out_b = out[b].reshape(-1)
            out_b[chosen] = True
        return out

    if mode == "sensor_fix":
        xy_map = mask.any(axis=0).any(axis=0)  
        flat = np.flatnonzero(xy_map)
        n_keep = int(round(flat.size * keep_ratio))
        chosen = _subsample_flat_indices(flat, n_keep)
        out_view = out.reshape(B, T, X * Y)
        out_view[:, :, chosen] = True
        return out

    if mode == "sensor_random":
        for b in range(B):
            xy_map_b = mask[b].any(axis=0)
            flat = np.flatnonzero(xy_map_b)
            n_keep = int(round(flat.size * keep_ratio))
            chosen = _subsample_flat_indices(flat, n_keep)
            out_view_b = out[b].reshape(T, X * Y)
            out_view_b[:, chosen] = True
        return out

    for b in range(B):
        flat = np.flatnonzero(mask[b])
        n_keep = int(round(flat.size * keep_ratio))
        chosen = _subsample_flat_indices(flat, n_keep)
        out_b = out[b].reshape(-1)
        out_b[chosen] = True
    return out

    
def fourier_wavenumbers(Nx: int, L: float = 1.0) -> np.ndarray:
    return 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)


def dealias_mask_23(Nx: int) -> np.ndarray:
    freq = np.fft.fftfreq(Nx) * Nx
    k_cut = Nx // 3
    return (np.abs(freq) <= k_cut).astype(np.float64)

def grf1_grid(
    *,
    N: int,
    m: float,
    gamma: float,
    tau: float,
    sigma: float,
    Nx: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    if Nx is None:
        Nx = 2 * N  

    my_const = 2.0 * np.pi
    j = np.arange(1, N + 1, dtype=np.float64)
    my_eigs = np.sqrt(2.0) * abs(sigma) * (((my_const * j) ** 2 + tau**2) ** (-gamma / 2.0))

    alpha = my_eigs * rng.standard_normal(N)
    beta = my_eigs * rng.standard_normal(N)

    x = np.linspace(0.0, 1.0, Nx, endpoint=False, dtype=np.float64)
    xs = x - 0.5 
    u = np.full_like(x, m, dtype=np.float64)
    for idx, jj in enumerate(j):
        u += alpha[idx] * np.cos(2.0 * np.pi * jj * xs) + beta[idx] * np.sin(2.0 * np.pi * jj * xs)
        
    return u.astype(np.float32)


def etdrk4_coeffs(k: np.ndarray, nu: float, dt: float):
    L = -nu * (k**2)
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / (LR**3), axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR**3), axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / (LR**3), axis=1))

    return E, E2, Q, f1, f2, f3


def nonlinear_term_hat(u_hat: np.ndarray, k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    u = np.fft.ifft(u_hat).real
    u2_hat = np.fft.fft(u * u)
    u2_hat *= mask  
    
    return -0.5j * k * u2_hat

def solve_burgers_1d(
    u0: np.ndarray,
    nu: float = 0.01,
    Nx: int = 128,
    Nt: int = 128,
    T: float = 1.0,
    L: float = 1.0,
    dealias: bool = True,
) -> np.ndarray:
    if u0.shape != (Nx,):
        raise ValueError(f"u0 shape must be ({Nx},), got {u0.shape}")

    dt = T / (Nt - 1)
    k = fourier_wavenumbers(Nx, L=L)
    mask = dealias_mask_23(Nx) if dealias else np.ones(Nx, dtype=np.float64)

    E, E2, Q, f1, f2, f3 = etdrk4_coeffs(k=k, nu=nu, dt=dt)
    u_hat = np.fft.fft(u0.astype(np.float64))

    U = np.zeros((Nt, Nx), dtype=np.float64)
    U[0] = u0

    for n in range(1, Nt):
        Nv = nonlinear_term_hat(u_hat, k, mask)
        a_hat = E2 * u_hat + Q * Nv
        Na = nonlinear_term_hat(a_hat, k, mask)
        b_hat = E2 * u_hat + Q * Na
        Nb = nonlinear_term_hat(b_hat, k, mask)
        c_hat = E2 * a_hat + Q * (2.0 * Nb - Nv)
        Nc = nonlinear_term_hat(c_hat, k, mask)

        u_hat = E * u_hat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        U[n] = np.fft.ifft(u_hat).real

    return U.astype(np.float32)

def generate_dataset(
    n_samples: int =40,
    nu: float = 0.01,
    Nx: int = 128,
    Nt: int = 128,
    T: float = 1.0,
    L: float = 1.0,
    seed: int = 0,
    U0: np.ndarray = None,
    warmup_steps: int = 10,
) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)

    u0_provided = U0 is not None
    if u0_provided:
        U0_arr = np.asarray(U0, dtype=np.float32)
    else:
        U0_arr = np.zeros((n_samples, 1, 1, Nx), dtype=np.float32)
    warmup = 0 if u0_provided else int(warmup_steps)

    Nt_sim = Nt + warmup
    U_full = np.zeros((n_samples, 1, Nt, Nx), dtype=np.float32)

    grf_N = Nx // 2

    for i in tqdm(range(n_samples)):
        if u0_provided:
            u0 = U0_arr[i, 0, 0].astype(np.float32, copy=False)
        else:
            u0 = grf1_grid(
                N=grf_N,
                m=0.0,
                gamma=2.0,
                tau=5.0,
                sigma=25.0,
                Nx=Nx,
                rng=rng,
            ).astype(np.float32, copy=False)
            U0_arr[i, 0, 0] = u0

        if Nt > 1:
            dt = T / (Nt - 1)
            T_sim = dt * (Nt_sim - 1)
        else:
            T_sim = T

        U_sim = solve_burgers_1d(
            u0=u0,
            nu=nu,
            Nx=Nx,
            Nt=Nt_sim,
            T=T_sim,
            L=L,
            dealias=True,
        )

        U_full[i, 0] = U_sim[warmup : warmup + Nt]

    return U_full