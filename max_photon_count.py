#!/usr/bin/env python3
"""
scatter_photon_upper_bound_dyn_gain.py
--------------------------------------
For 10,000 random pixels in a webcam recording, fit Gaussian per-channel,
compute max photon count from μ²/σ². Print full summary table every 100
pixels with best μ, σ, N_max per channel.
"""

from __future__ import annotations
import random, numpy as np, matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────
PATH       = "webcam_1000_frames.npz"
N_PIXELS   = 10_000
BLOCK      = 100                      # report interval
BIN_MIN, BIN_MAX = 0, 255
EPS        = 1e-12                    # avoid div-by-zero
# -------------------------------------------------------------------------

def gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-6)  # clamp to avoid divide-by-zero
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
    frames = np.load(arch)["frames"]           # (N,H,W,3) uint8
    n_frames, H, W, _ = frames.shape
    print(f"{n_frames} frames, {W}×{H} px")

    coords = random_unique_coords(H, W, N_PIXELS)
    print(f"Selected {N_PIXELS} random pixel locations")

    mus     = {c: np.full(N_PIXELS, np.nan) for c in "RGB"}
    sigmas  = {c: np.full(N_PIXELS, np.nan) for c in "RGB"}
    nmaxs   = {c: np.full(N_PIXELS, np.nan) for c in "RGB"}  # μ² / σ²

    for idx, (y, x) in enumerate(coords):
        b = frames[:, y, x, 0].astype(float)  # BGR order
        g = frames[:, y, x, 1].astype(float)
        r = frames[:, y, x, 2].astype(float)

        for ch, data in zip(("R", "G", "B"), (r, g, b)):
            mu0, s0 = data.mean(), data.std(ddof=0)
            mu_fit, s_fit = fit_gauss(data, mu0, s0)

            # Filter unreasonable values
            if mu_fit > 245 or mu_fit < 10 or s_fit < 0.01:
                continue

            mus[ch][idx]    = mu_fit
            sigmas[ch][idx] = s_fit
            nmaxs[ch][idx]  = (mu_fit ** 2) / (s_fit ** 2 + EPS)

        # ── block report every 100 pixels ────────────────────────────────
        if (idx + 1) % BLOCK == 0:
            header = "{:<3}  {:>12}  {:>10}  {:>10}".format("CH", "N_max", "μ", "σ")
            report = [header, "-" * len(header)]
            for ch in "RGB":
                valid_mask = ~np.isnan(nmaxs[ch][:idx + 1])
                if not np.any(valid_mask):
                    report.append(f"{ch:<3}  {'(no data)':>12}  {'–':>10}  {'–':>10}")
                else:
                    imax = np.nanargmax(nmaxs[ch][:idx + 1])
                    line = "{:<3}  {:12.3e}  {:10.3f}  {:10.3f}".format(
                        ch,
                        nmaxs[ch][imax],
                        mus[ch][imax],
                        sigmas[ch][imax]
                    )
                    report.append(line)
            print(f"[{idx + 1:5d}/{N_PIXELS}] max N –")
            print("\n".join(report))

    # ── final scatter plots ──────────────────────────────────────────────
    for ch, colour in (("R", "red"), ("G", "green"), ("B", "blue")):
        plt.figure(f"{ch} channel   μ vs N_max")
        plt.scatter(mus[ch], nmaxs[ch], c=colour, s=10, alpha=0.6)
        plt.yscale("log")
        plt.xlabel("Fitted mean μ (DN)")
        plt.ylabel("Maximum photon count N_max")
        plt.title(f"Shot-noise upper bound – {ch} channel")
        plt.grid(True, ls="--", lw=0.4, alpha=0.5)
        plt.tight_layout()

    print("\nFinal maximum N_max per channel:")
    for ch in "RGB":
        if np.all(np.isnan(nmaxs[ch])):
            print(f"  {ch}: no valid data")
        else:
            imax = np.nanargmax(nmaxs[ch])
            print(f"  {ch}: {nmaxs[ch][imax]:.3e} photons  (μ={mus[ch][imax]:.3f}, σ={sigmas[ch][imax]:.3f})")

    plt.show()

if __name__ == "__main__":
    main()
