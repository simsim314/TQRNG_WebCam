#!/usr/bin/env python3
"""
scatter_mu_sigma_1000px.py
--------------------------
Load “webcam_1000_frames.npz”, randomly pick 1 000 pixel locations, fit a
Gaussian N(μ,σ²) to the per-frame intensity series of EACH pixel and EACH
colour channel, and finally create **three separate scatter plots**:

    •   Figure 1 – Red channel   (μ_R  vs.  σ_R)
    •   Figure 2 – Green channel (μ_G  vs.  σ_G)
    •   Figure 3 – Blue channel  (μ_B  vs.  σ_B)

Dependencies: numpy, matplotlib, opencv-python; SciPy (optional) for curve_fit.
"""
from __future__ import annotations
import random, numpy as np, matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path

PATH = "webcam_1000_frames.npz"
#PATH = "webcam_recording1.npz"
# ── configuration ─────────────────────────────────────────────────────────
ARCHIVE   = Path(PATH)
N_PIXELS  = 10_000
BIN_MIN, BIN_MAX = 0, 255
# -------------------------------------------------------------------------

def gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

def fit_gauss(samples: np.ndarray, mu0: float, sigma0: float) -> tuple[float,float]:
    """Fit a Gaussian to the 8-bit histogram; fall back to grid-search if SciPy absent."""
    x_bins = np.arange(BIN_MIN, BIN_MAX+1)
    hist, _ = np.histogram(samples, np.arange(BIN_MIN-0.5, BIN_MAX+1.5), density=True)
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(gauss, x_bins, hist, p0=[mu0, sigma0],
                            bounds=([0.0, 1e-3], [255.0, 500.0]), maxfev=10_000)
        return float(popt[0]), float(popt[1])
    except Exception:
        mu_grid    = np.linspace(mu0-5,  mu0+5,  41)
        sigma_grid = np.linspace(max(0.5, 0.5*sigma0), 2*sigma0, 41)
        best, err  = (mu0, sigma0), np.inf
        for mu in mu_grid:
            for s in sigma_grid:
                e = np.mean((gauss(x_bins, mu, s) - hist)**2)
                if e < err: best, err = (mu, s), e
        return best

def random_unique_coords(h: int, w: int, n: int) -> list[tuple[int,int]]:
    """Return *n* unique (y,x) coordinates inside an h×w image."""
    coords = set()
    while len(coords) < n:
        coords.add((random.randint(0, h-1), random.randint(0, w-1)))
    return list(coords)

def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError("Record first – webcam_1000_frames.npz not present")

    print(f"Loading {ARCHIVE} …")
    frames = np.load(ARCHIVE)["frames"]            # (N,H,W,3)   dtype uint8
    n_frames, H, W, _ = frames.shape
    print(f"{n_frames} frames, {W}×{H} px")

    coords = random_unique_coords(H, W, N_PIXELS)
    print(f"Selected {N_PIXELS} random pixel locations")

    # containers for fitted parameters
    mus  = {c: np.empty(N_PIXELS) for c in "RGB"}
    sigs = {c: np.empty(N_PIXELS) for c in "RGB"}

    for idx, (y, x) in enumerate(coords):
        b = frames[:, y, x, 0].astype(float)   # BGR order in stored frames
        g = frames[:, y, x, 1].astype(float)
        r = frames[:, y, x, 2].astype(float)

        for ch, data in zip(("R","G","B"), (r, g, b)):
            mu0, s0            = data.mean(), data.std(ddof=0)
            mu_fit, s_fit      = fit_gauss(data, mu0, s0)
            mus[ch][idx]       = mu_fit
            sigs[ch][idx]      = s_fit

        if (idx+1) % 100 == 0 or idx == 0:
            print(f"  fitted {idx+1}/{N_PIXELS} pixels")

    # ── separate scatter figures ─────────────────────────────────────────
    channel_info = [("R", "red"), ("G", "green"), ("B", "blue")]
    for ch, colour in channel_info:
        plt.figure(f"{ch} channel   μ vs σ")
        plt.scatter(mus[ch], sigs[ch], c=colour, s=10, alpha=0.6)
        plt.xlabel("Fitted μ (mean value)")
        plt.ylabel("Fitted σ (standard deviation)")
        plt.title(f"Gaussian fit parameters – {ch} channel")
        plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
