from __future__ import annotations

import argparse
import numpy as np
from tqdm import tqdm
import torch

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

def make_sensor_mask(Nx: int, Nt: int, n_sensors: int = None, sparse_ratio: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    if n_sensors is not None:
        x_idx = rng.choice(Nx, size=n_sensors, replace=False)
        mask = np.zeros((Nt, Nx), dtype=bool)
        mask[:, x_idx] = True
    else:
        n = int(round(Nt * Nx * sparse_ratio))
        flat_idx = rng.choice(Nt * Nx, size=n, replace=False)
        t_idx, x_idx = np.unravel_index(flat_idx, (Nt, Nx))
        mask = np.zeros((Nt, Nx), dtype=bool)
        mask[t_idx, x_idx] = True
        
    return mask

def apply_spatial_mask(U: np.ndarray, mask: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    U_obs = np.full_like(U, fill_value, dtype=np.float32)
    U_obs[mask] = U[mask]
    
    return U_obs
import numpy as np

def sample_u0_burgers_paper(
    Nx: int,
    L: float = 1.0,
    alpha: float = 25.0,      # (-Δ + alpha I)
    amp: float = 625.0,       # overall scale
    rng: np.random.Generator | None = None,
    remove_mean: bool = True,
) -> np.ndarray:
    """
    u0 ~ N(0, amp * (-Δ + alpha I)^(-2)) on 1D torus of length L.

    Returns:
        u0: (Nx,) float32, periodic on grid x in [0, L)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Fourier mode indices consistent with np.fft conventions
    k = np.fft.fftfreq(Nx, d=L / Nx)  # cycles per unit length
    lam = (2.0 * np.pi * k) ** 2      # eigenvalues of (-Δ) on torus

    # Covariance eigenvalues for (-Δ + alpha I)^(-2)
    # => spectral density ~ 1/(lam + alpha)^2
    S = amp / (lam + alpha) ** 2

    # Sample complex Fourier coefficients with the desired variance.
    # For a real field: enforce Hermitian symmetry by constructing in frequency domain.
    z = rng.standard_normal(Nx) + 1j * rng.standard_normal(Nx)
    uhat = np.sqrt(S / 2.0) * z

    # Enforce real-valued signal constraints for rFFT-compatible symmetry
    uhat[0] = np.sqrt(S[0]) * rng.standard_normal() + 0j
    if Nx % 2 == 0:
        uhat[Nx // 2] = np.sqrt(S[Nx // 2]) * rng.standard_normal() + 0j
    # Hermitian symmetry: uhat[-k] = conj(uhat[k])
    for i in range(1, (Nx // 2) if Nx % 2 == 0 else (Nx + 1) // 2):
        uhat[-i] = np.conj(uhat[i])

    u0 = np.fft.ifft(uhat).real

    if remove_mean:
        u0 = u0 - u0.mean()

    return u0.astype(np.float32)

def generate_dataset(
    n_samples: int,
    nu: float = 0.01,
    Nx: int = 128,
    Nt: int = 128,
    T: float = 1.0,
    L: float = 1.0,
    grf_m: float = 0.0,
    grf_gamma: float = 2.0,
    grf_tau: float = 5.0,
    grf_sigma: float = 25.0,
    n_sensors: int = 5,
    sprse_ratio: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, L, Nx, endpoint=False).astype(np.float32)
    t = np.linspace(0.0, T, Nt).astype(np.float32)

    U_full = np.zeros((n_samples, 1, Nt, Nx), dtype=np.float32)
    U_obs  = np.zeros((n_samples, 1, Nt, Nx), dtype=np.float32)
    U0     = np.zeros((n_samples, 1, 1, Nx), dtype=np.float32)

    grf_N = Nx // 2

    if n_sensors:
        mask = make_sensor_mask(Nx, Nt, n_sensors=n_sensors, rng=np.random.default_rng(seed+1))
    else:
        mask = make_sensor_mask(Nx, Nt, sparse_ratio=sprse_ratio, rng=np.random.default_rng(seed+1))

    for i in tqdm(range(n_samples)):
        u0 = grf1_grid(
            N=grf_N,
            m=grf_m,
            gamma=grf_gamma,
            tau=grf_tau,
            sigma=grf_sigma,
            Nx=Nx,
            rng=rng,
        )

        U = solve_burgers_1d(
            u0=u0,
            nu=nu,
            Nx=Nx,
            Nt=Nt,
            T=T,
            L=L,
            dealias=True,
        )

        U_full[i, 0] = U
        U0[i, 0, 0]  = u0

        U_obs_i = apply_spatial_mask(U, mask)
        U_obs[i, 0] = U_obs_i

    return {
        "x": x,
        "t": t,
        "nu": np.asarray(nu, dtype=np.float32),
        "U_full": U_full,  
        "U_obs": U_obs,    
        "u0": U0,      
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
    sprse_ratio: float = 0.1,
    seed: int = 0,
) :

    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, L, Nx, endpoint=False).astype(np.float32)
    t = np.linspace(0.0, T, Nt).astype(np.float32)

    U_full = np.zeros((n_samples, 1, Nt, Nx), dtype=np.float32)
    U_obs  = np.zeros((n_samples, 1, Nt, Nx), dtype=np.float32)
    U0     = np.zeros((n_samples, 1, 1, Nx), dtype=np.float32)

    if n_sensors:
        mask = make_sensor_mask(Nx, Nt, n_sensors=n_sensors, rng=np.random.default_rng(seed+1))
    else:
        mask = make_sensor_mask(Nx, Nt, sparse_ratio=sprse_ratio, rng=np.random.default_rng(seed+1))

    for i in tqdm(range(n_samples)):
        u0 = predict[0,:].detach().cpu().numpy()
        U = solve_burgers_1d(
            u0=u0,
            nu=nu,
            Nx=Nx,
            Nt=Nt,
            T=T,
            L=L,
            dealias=True,
        )

        U_full[i, 0] = U
        U0[i, 0, 0]  = u0

        U_obs_i = apply_spatial_mask(U, mask)
        U_obs[i, 0] = U_obs_i

    return {
        "x": x,
        "t": t,
        "nu": np.asarray(nu, dtype=np.float32),
        "U_full": U_full,  
        "U_obs": U_obs,   
        "u0": U0,      
        "mask": mask,
    }