#!/usr/bin/env python3
"""
scatter_brightness_upper_bound.py
---------------------------------
For 10,000 random pixels in a Y-only webcam recording, fit a Gaussian,
and compute the max photon count estimate from μ²/σ².

Reports max μ, σ, N_max, min σ and σ at max N_max every 100 pixels.
"""

import random, numpy as np, matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────
PATH       = "webcam_1000_y_frames.npz"
N_PIXELS   = 10_000
BLOCK      = 100
BIN_MIN, BIN_MAX = 0, 255
EPS        = 1e-12  # avoid div-by-zero
# --------------------------------------------------------------------------

def gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-6)
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)

def fit_gauss(samples: np.ndarray, mu0: float, sigma0: float) -> tuple[float,float]:
    x_bins = np.arange(BIN_MIN, BIN_MAX + 1)
    hist, _ = np.histogram(samples,
                           np.arange(BIN_MIN - 0.5, BIN_MAX + 1.5),
                           density=True)
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(gauss, x_bins, hist, p0=[mu0, sigma0],
                            bounds=([0.0, 1e-3], [255.0, 500.0]),
                            maxfev=10_000)
        return float(popt[0]), float(popt[1])
    except Exception:
        mu_grid    = np.linspace(mu0 - 5, mu0 + 5, 41)
        sigma_grid = np.linspace(max(0.5, 0.5 * sigma0), 2 * sigma0, 41)
        best, err  = (mu0, sigma0), np.inf
        for mu in mu_grid:
            for s in sigma_grid:
                e = np.mean((gauss(x_bins, mu, s) - hist)**2)
                if e < err:
                    best, err = (mu, s), e
        return best

def random_unique_coords(h: int, w: int, n: int) -> list[tuple[int, int]]:
    coords = set()
    while len(coords) < n:
        coords.add((random.randint(0, h - 1), random.randint(0, w - 1)))
    return list(coords)

def main() -> None:
    arch = Path(PATH)
    if not arch.exists():
        raise FileNotFoundError(f"{arch} not found – record first!")

    print(f"Loading {arch} …")
    frames = np.load(arch)["y_frames"]  # shape: (N, H, W), dtype=uint8
    n_frames, H, W = frames.shape
    print(f"{n_frames} Y-frames, {W}×{H} px")

    coords = random_unique_coords(H, W, N_PIXELS)
    print(f"Selected {N_PIXELS} random pixel locations")

    mus    = np.full(N_PIXELS, np.nan)
    sigmas = np.full(N_PIXELS, np.nan)
    nmaxs  = np.full(N_PIXELS, np.nan)

    for idx, (y, x) in enumerate(coords):
        data = frames[:, y, x].astype(float)
        mu0, s0 = data.mean(), data.std(ddof=0)
        mu_fit, s_fit = fit_gauss(data, mu0, s0)

        if mu_fit > 245 or mu_fit < 10 or s_fit < 0.01:
            continue

        mus[idx]    = mu_fit
        sigmas[idx] = s_fit
        nmaxs[idx]  = (mu_fit ** 2) / (s_fit ** 2 + EPS)

        # ── Report every BLOCK pixels ─────────────────────────────────
        if (idx + 1) % BLOCK == 0:
            valid_mask = ~np.isnan(nmaxs[:idx + 1])
            if np.any(valid_mask):
                imax = np.nanargmax(nmaxs[:idx + 1])
                min_sigma = np.nanmin(sigmas[:idx + 1])
                report = (
                    f"[{idx + 1:5d}/{N_PIXELS}] stats:\n"
                    f"  Max N_max = {nmaxs[imax]:.3e}\n"
                    f"     μ      = {mus[imax]:.3f}\n"
                    f"     σ      = {sigmas[imax]:.3f} (at max N_max)\n"
                    f"  Min σ so far = {min_sigma:.4f}\n"
                )
                print(report)
            else:
                print(f"[{idx + 1:5d}/{N_PIXELS}] No valid data yet")

    # ── Plot μ vs N_max ──────────────────────────────────────────────────
    plt.figure("Brightness μ vs N_max")
    plt.scatter(mus, nmaxs, c="black", s=10, alpha=0.6)
    plt.yscale("log")
    plt.xlabel("Fitted mean μ (DN)")
    plt.ylabel("Maximum photon count N_max")
    plt.title("Shot-noise upper bound – brightness only")
    plt.grid(True, ls="--", lw=0.4, alpha=0.5)
    plt.tight_layout()

    # ── Plot μ vs σ ───────────────────────────────────────────────────────
    plt.figure("Brightness μ vs σ")
    plt.scatter(mus, sigmas, c="blue", s=10, alpha=0.6)
    plt.xlabel("Fitted mean μ (DN)")
    plt.ylabel("Fitted σ (DN)")
    plt.title("Pixel noise (σ) vs brightness (μ)")
    plt.grid(True, ls="--", lw=0.4, alpha=0.5)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
