"""
Burgers' equation dataset generator (DiffusionPDE/FNO-style)
- PDE solver: Fourier pseudo-spectral + ETDRK4 + 2/3 dealias
- Initial condition: GRF implemented to match MATLAB GRF1.m (alpha/beta trig series)
- Output: .npz with U_full / U_obs / mask / x / t / nu / u0

Burgers equation:
    ∂_t u(x,t) + ∂_x (u(x,t)^2 / 2) = ν ∂_{xx} u(x,t)
    x ∈ (0,1), t ∈ (0,T], periodic BC for the solver.

Default "paper" grid:
    Nx = 128
    Nt = 128 (u0 + 127 steps)
    T  = 1.0

Run example (paper-like):
    python burgers_grf1_dataset.py --out burgers_nu001.npz --n_samples 50000 --nu 0.01 --fixed_sensors

Time-dense example:
    python burgers_grf1_dataset.py --out burgers_Nt1024.npz --n_samples 4 --nu 0.01 --Nx 128 --Nt 1024 --T 1.0 --fixed_sensors
"""

from __future__ import annotations

import argparse
import numpy as np
from tqdm import tqdm

# -----------------------------
# Utilities: Fourier grids
# -----------------------------
def fourier_wavenumbers(Nx: int, L: float = 1.0) -> np.ndarray:
    """Angular wavenumbers k for periodic domain length L."""
    return 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)


def dealias_mask_23(Nx: int) -> np.ndarray:
    """2/3-rule dealiasing mask in FFT ordering."""
    freq = np.fft.fftfreq(Nx) * Nx
    k_cut = Nx // 3
    return (np.abs(freq) <= k_cut).astype(np.float64)


# -----------------------------
# GRF initial condition (MATLAB GRF1.m equivalent) -> grid values
# -----------------------------
def grf1_grid(
    *,
    N: int,
    m: float,
    gamma: float,
    tau: float,
    sigma: float,
    type: str = "periodic",
    Nx: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Python port of the user's MATLAB GRF1.m, returning u0 on a grid.

    GRF1.m:
      Random function from N(m, C) on [0,1] where
        C = sigma^2 (-Delta + tau^2 I)^(-gamma)
      with periodic / zero Dirichlet / zero Neumann boundary.
      Dirichlet only supports m = 0.
      N is # of Fourier modes (usually grid_size/2).
      periodic case includes a shift: uu(t - 0.5)

    This function mirrors that logic:
      - periodic: u(x)= m + Σ [α_j cos(2π j (x-0.5)) + β_j sin(2π j (x-0.5))]
      - dirichlet: u(x)= Σ β_j sin(π j x), with m forced to 0
      - neumann:   u(x)= m + Σ α_j cos(π j x)

    NOTE:
      For the Burgers solver here, the PDE BC is periodic. If you choose non-periodic
      GRF types ("dirichlet"/"neumann"), you will still simulate Burgers with periodic BC.
      For exact match to DiffusionPDE/FNO Burgers benchmark, use type="periodic" and m=0.
    """
    if rng is None:
        rng = np.random.default_rng()
    type = type.lower().strip()
    if type not in {"periodic", "dirichlet", "neumann"}:
        raise ValueError("type must be one of: 'periodic', 'dirichlet', 'neumann'")

    # Match GRF1.m: Dirichlet forces m = 0
    if type == "dirichlet":
        m = 0.0

    if Nx is None:
        Nx = 2 * N  # GRF1.m comment: N is usually grid_size/2

    # Match GRF1.m: periodic uses 2*pi, otherwise pi
    my_const = 2.0 * np.pi if type == "periodic" else np.pi

    j = np.arange(1, N + 1, dtype=np.float64)

    # Match GRF1.m exactly:
    # my_eigs = sqrt(2) * abs(sigma) * ((my_const*j)^2 + tau^2)^(-gamma/2)
    my_eigs = np.sqrt(2.0) * abs(sigma) * (((my_const * j) ** 2 + tau**2) ** (-gamma / 2.0))

    # Sample alpha, beta with BC-specific zeroing (matches GRF1.m)
    if type == "dirichlet":
        alpha = np.zeros(N, dtype=np.float64)
    else:
        alpha = my_eigs * rng.standard_normal(N)

    if type == "neumann":
        beta = np.zeros(N, dtype=np.float64)
    else:
        beta = my_eigs * rng.standard_normal(N)

    # Build grid and evaluate series
    if type == "periodic":
        # endpoint=False avoids duplicating x=1 which equals x=0 under periodicity
        x = np.linspace(0.0, 1.0, Nx, endpoint=False, dtype=np.float64)
        xs = x - 0.5  # GRF1.m: uu(t-0.5)
        u = np.full_like(x, m, dtype=np.float64)
        # u = m + Σ [α cos(2π j (x-0.5)) + β sin(2π j (x-0.5))]
        for idx, jj in enumerate(j):
            u += alpha[idx] * np.cos(2.0 * np.pi * jj * xs) + beta[idx] * np.sin(2.0 * np.pi * jj * xs)
        return u.astype(np.float32)
    
    # if type == "periodic":
    #     # endpoint=False avoids duplicating x=1 which equals x=0 under periodicity
    #     x = np.linspace(-1, 1.0, Nx, endpoint=False, dtype=np.float64)
    #     u = -np.sin(np.pi*x)
    #     return u.astype(np.float32)
    
    elif type == "dirichlet":
        # sine basis on [0,1], mean forced 0
        x = np.linspace(0.0, 1.0, Nx, endpoint=True, dtype=np.float64)
        u = np.zeros_like(x, dtype=np.float64)
        for idx, jj in enumerate(j):
            u += beta[idx] * np.sin(np.pi * jj * x)
        return u.astype(np.float32)

    else:  # neumann
        # cosine basis on [0,1], include mean m
        x = np.linspace(0.0, 1.0, Nx, endpoint=True, dtype=np.float64)
        u = np.full_like(x, m, dtype=np.float64)
        for idx, jj in enumerate(j):
            u += alpha[idx] * np.cos(np.pi * jj * x)
        return u.astype(np.float32)


# -----------------------------
# Burgers solver: ETDRK4 + pseudo-spectral
# -----------------------------
def etdrk4_coeffs(k: np.ndarray, nu: float, dt: float):
    """
    Precompute ETDRK4 coefficients for linear operator L = -nu*k^2.
    (k is angular wavenumber)
    """
    L = -nu * (k**2)
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    # Contour integral approximation (Kassam & Trefethen style)
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / (LR**3), axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR**3), axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / (LR**3), axis=1))

    return E, E2, Q, f1, f2, f3


def nonlinear_term_hat(u_hat: np.ndarray, k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Nonlinear term in Fourier space:
        N(u) = -∂_x (u^2 / 2) = -(1/2) ∂_x (u^2)
    """
    u = np.fft.ifft(u_hat).real
    u2_hat = np.fft.fft(u * u)
    u2_hat *= mask  # dealias
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
    """
    Solve Burgers on periodic domain (0,L), output u(t_n, x_j) with Nt frames.
    Convention matches paper: u0 + 127 time steps => Nt=128, dt=T/(Nt-1)
    """
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


# -----------------------------
# Mask/sensor simulation
# -----------------------------
def make_sensor_mask(Nx: int, n_sensors: int = 5, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    idx = rng.choice(Nx, size=n_sensors, replace=False)
    mask = np.zeros(Nx, dtype=bool)
    mask[idx] = True
    return mask


def apply_spatial_mask(U: np.ndarray, mask: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """U: (Nt, Nx) -> observed version with only masked x kept across all time."""
    U_obs = np.full_like(U, fill_value, dtype=np.float32)
    U_obs[:, mask] = U[:, mask]
    return U_obs


# -----------------------------
# Dataset builder
# -----------------------------
def generate_dataset(
    n_samples: int,
    nu: float = 0.01,
    Nx: int = 128,
    Nt: int = 128,
    T: float = 1.0,
    L: float = 1.0,
    # GRF1 params
    grf_N: int | None = None,
    grf_m: float = 0.0,
    grf_gamma: float = 2.0,
    grf_tau: float = 5.0,
    grf_sigma: float = 25.0,
    grf_type: str = "periodic",
    # sensors
    n_sensors: int = 5,
    fixed_sensors: bool = False,
    seed: int = 0,
) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, L, Nx, endpoint=False).astype(np.float32)
    t = np.linspace(0.0, T, Nt).astype(np.float32)

    # 🔑 이미지 텐서로 바로 생성
    U_full = np.zeros((n_samples, 1, Nt, Nx), dtype=np.float32)
    U_obs  = np.zeros((n_samples, 1, Nt, Nx), dtype=np.float32)
    U0     = np.zeros((n_samples, 1, 1, Nx), dtype=np.float32)

    if grf_N is None:
        grf_N = Nx // 2

    if fixed_sensors:
        mask = make_sensor_mask(Nx, n_sensors=n_sensors, rng=np.random.default_rng(seed+1))
    else:
        mask = np.zeros((n_samples, Nx), dtype=bool)

    for i in tqdm(range(n_samples)):
        # ---- Initial condition (1D field)
        u0 = grf1_grid(
            N=grf_N,
            m=grf_m,
            gamma=grf_gamma,
            tau=grf_tau,
            sigma=grf_sigma,
            type=grf_type,
            Nx=Nx,
            rng=rng,
        )

        # ---- Burgers solve → (Nt, Nx)
        U = solve_burgers_1d(
            u0=u0,
            nu=nu,
            Nx=Nx,
            Nt=Nt,
            T=T,
            L=L,
            dealias=True,
        )

        # ---- Store as images
        U_full[i, 0] = U
        U0[i, 0, 0]  = u0

        if fixed_sensors:
            U_obs_i = apply_spatial_mask(U, mask)
        else:
            msk = make_sensor_mask(Nx, n_sensors=n_sensors, rng=rng)
            mask[i] = msk
            U_obs_i = apply_spatial_mask(U, msk)

        # NaN 유지 (or 여기서 0으로 바꿔도 됨)
        U_obs[i, 0] = U_obs_i

    return {
        "x": x,
        "t": t,
        "nu": np.asarray(nu, dtype=np.float32),
        "U_full": U_full,   # (N, 1, Nt, Nx)
        "U_obs": U_obs,     # (N, 1, Nt, Nx)
        "u0": U0,           # (N, 1, 1, Nx)
        "mask": mask,
        "grf_params": np.asarray(
            [grf_N, grf_m, grf_gamma, grf_tau, grf_sigma],
            dtype=np.float32,
        ),
    }

def generate_dataset_predict_initial(
    n_samples: int,
    nu: float = 0.01,
    Nx: int = 128,
    Nt: int = 128,
    T: float = 1.0,
    L: float = 1.0,
    predict: torch.tensor = None,
    n_sensors: int = 5,
    fixed_sensors: bool = False,
    seed: int = 0,
) :

    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, L, Nx, endpoint=False).astype(np.float32)
    t = np.linspace(0.0, T, Nt).astype(np.float32)

    # 🔑 이미지 텐서로 바로 생성
    U_full = np.zeros((n_samples, 1, Nt, Nx), dtype=np.float32)
    U_obs  = np.zeros((n_samples, 1, Nt, Nx), dtype=np.float32)
    U0     = np.zeros((n_samples, 1, 1, Nx), dtype=np.float32)

    if fixed_sensors:
        mask = make_sensor_mask(Nx, n_sensors=n_sensors, rng=np.random.default_rng(seed+1))
    else:
        mask = np.zeros((n_samples, Nx), dtype=bool)

    for i in tqdm(range(n_samples)):
        # ---- Initial condition (1D field)
        u0 = predict[0,:].detach().cpu().numpy()
        # ---- Burgers solve → (Nt, Nx)
        U = solve_burgers_1d(
            u0=u0,
            nu=nu,
            Nx=Nx,
            Nt=Nt,
            T=T,
            L=L,
            dealias=True,
        )

        # ---- Store as images
        U_full[i, 0] = U
        U0[i, 0, 0]  = u0

        if fixed_sensors:
            U_obs_i = apply_spatial_mask(U, mask)
        else:
            msk = make_sensor_mask(Nx, n_sensors=n_sensors, rng=rng)
            mask[i] = msk
            U_obs_i = apply_spatial_mask(U, msk)

        # NaN 유지 (or 여기서 0으로 바꿔도 됨)
        U_obs[i, 0] = U_obs_i

    return {
        "x": x,
        "t": t,
        "nu": np.asarray(nu, dtype=np.float32),
        "U_full": U_full,   # (N, 1, Nt, Nx)
        "U_obs": U_obs,     # (N, 1, Nt, Nx)
        "u0": U0,           # (N, 1, 1, Nx)
        "mask": mask,
    }

def save_npz(out_path: str, data: dict[str, np.ndarray]):
    np.savez_compressed(out_path, **data)


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser()

    # Output / samples
    p.add_argument("--out", type=str, default="burgers_grf1_like.npz")
    p.add_argument("--n_samples", type=int, default=1000)

    # PDE
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--Nx", type=int, default=128)
    p.add_argument("--Nt", type=int, default=128)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--L", type=float, default=1.0)

    # GRF1 (MATLAB-equivalent parameters)
    p.add_argument("--grf_N", type=int, default=None, help="Number of modes (default: Nx//2).")
    p.add_argument("--grf_m", type=float, default=0.0)
    p.add_argument("--grf_gamma", type=float, default=2.0)
    p.add_argument("--grf_tau", type=float, default=5.0)
    p.add_argument("--grf_sigma", type=float, default=25.0)
    p.add_argument("--grf_type", type=str, default="periodic", choices=["periodic", "dirichlet", "neumann"])

    # Sensors / masking
    p.add_argument("--n_sensors", type=int, default=5)
    p.add_argument("--fixed_sensors", action="store_true")

    # RNG
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    data = generate_dataset(
        n_samples=args.n_samples,
        nu=args.nu,
        Nx=args.Nx,
        Nt=args.Nt,
        T=args.T,
        L=args.L,
        grf_N=args.grf_N,
        grf_m=args.grf_m,
        grf_gamma=args.grf_gamma,
        grf_tau=args.grf_tau,
        grf_sigma=args.grf_sigma,
        grf_type=args.grf_type,
        n_sensors=args.n_sensors,
        fixed_sensors=args.fixed_sensors,
        seed=args.seed,
    )

    save_npz(args.out, data)
    print(f"Saved dataset to: {args.out}")
    print(f"U_full shape: {data['U_full'].shape}  (n_samples, Nt, Nx)")
    print(f"nu={float(data['nu'])}  Nx={args.Nx}  Nt={args.Nt}  T={args.T}")
    print(f"GRF1 params: N={args.grf_N if args.grf_N is not None else args.Nx//2}, m={args.grf_m}, "
          f"gamma={args.grf_gamma}, tau={args.grf_tau}, sigma={args.grf_sigma}, type={args.grf_type}")
    if args.fixed_sensors:
        print(f"Sensors: fixed, n_sensors={args.n_sensors}")
    else:
        print(f"Sensors: per-sample random, n_sensors={args.n_sensors}")


if __name__ == "__main__":
    main()
