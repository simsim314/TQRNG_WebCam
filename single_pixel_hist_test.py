#!/usr/bin/env python3
"""
single_pixel_gauss_fit_thresh_from_npz.py
-----------------------------------------
Load a recorded frame stack (“webcam_1000_frames.npz”), choose one pixel
whose brightness lies inside the window

    LOW_THRESH < max(R,G,B) < SAT_LIMIT,

gather that pixel’s samples across every frame, fit a Gaussian to the
distribution in each colour channel, and plot the resulting histograms and
curves.

Dependencies
------------
• numpy, matplotlib, opencv-python
• (optional) scipy  – provides curve_fit; script falls back to grid search
                      if SciPy is not present.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── configuration ─────────────────────────────────────────────────────────
ARCHIVE = Path("webcam_1000_frames.npz")   # change path if needed
BIN_MIN, BIN_MAX = 0, 255                  # histogram support

LOW_THRESH = 30      # lower brightness bound
SAT_LIMIT  = 90      # upper brightness bound (avoid saturation)
# -------------------------------------------------------------------------


def pick_pixel(frame: np.ndarray, tries: int = 100_000) -> tuple[int, int]:
    """
    Randomly search for (y, x) such that

        LOW_THRESH < max(R, G, B) < SAT_LIMIT.

    Raises
    ------
    RuntimeError
        If no qualifying pixel is found after `tries` attempts.
    """
    h, w = frame.shape[:2]
    for _ in range(tries):
        y, x = random.randint(0, h - 1), random.randint(0, w - 1)
        b, g, r = frame[y, x]               # BGR order from OpenCV
        if LOW_THRESH < max(r, g, b) < SAT_LIMIT:
            return y, x
    raise RuntimeError(
        f"No pixel satisfying {LOW_THRESH} < max(R,G,B) < {SAT_LIMIT} found"
    )


def gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Standard normal PDF parameterised by mean μ and σ."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gauss(samples: np.ndarray, mu0: float, sigma0: float) -> tuple[float, float]:
    """
    Fit a Gaussian to an 8-bit histogram.  Tries ``scipy.optimize.curve_fit`` first;
    if SciPy is missing, falls back to a coarse grid search.
    """
    x_bins = np.arange(BIN_MIN, BIN_MAX + 1)
    hist, _ = np.histogram(
        samples, np.arange(BIN_MIN - 0.5, BIN_MAX + 1.5), density=True
    )

    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            gauss,
            x_bins,
            hist,
            p0=[mu0, sigma0],
            bounds=([0.0, 1e-3], [255.0, 500.0]),
            maxfev=10_000,
        )
        return float(popt[0]), float(popt[1])

    except Exception:
        # SciPy absent or fit failed → coarse grid search
        mu_grid = np.linspace(mu0 - 5, mu0 + 5, 41)
        sigma_grid = np.linspace(max(0.5, 0.5 * sigma0), 2 * sigma0, 41)
        best, err = (mu0, sigma0), np.inf
        for mu in mu_grid:
            for s in sigma_grid:
                e = np.mean((gauss(x_bins, mu, s) - hist) ** 2)
                if e < err:
                    best, err = (mu, s), e
        return best


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"{ARCHIVE} not found – record first!")

    print(f"Loading {ARCHIVE} …")
    frames = np.load(ARCHIVE)["frames"]          # shape (N, H, W, 3)
    n_frames, H, W, _ = frames.shape
    print(f"{n_frames} frames, {W}×{H} each")

    # ── choose pixel & extract samples ────────────────────────────────────
    y, x = pick_pixel(frames[0])
    print(f"Selected pixel (row={y}, col={x}); extracting samples …")

    chan = {c: [] for c in "RGB"}
    for frame in frames:
        b, g, r = frame[y, x]                   # BGR
        chan["R"].append(int(r))
        chan["G"].append(int(g))
        chan["B"].append(int(b))

    # ── plot ──────────────────────────────────────────────────────────────
    colours = {"R": "red", "G": "green", "B": "blue"}
    x_plot = np.linspace(BIN_MIN, BIN_MAX, 512)

    plt.figure(f"Gaussian fit – pixel ({y},{x})")

    for c in "RGB":
        data = np.asarray(chan[c], float)
        mu0, s0 = data.mean(), data.std(ddof=0)
        mu, s = fit_gauss(data, mu0, s0)

        hist, edges = np.histogram(
            data, np.arange(BIN_MIN - 0.5, BIN_MAX + 1.5), density=True
        )
        centres = 0.5 * (edges[:-1] + edges[1:])
        plt.step(centres, hist, where="mid", color=colours[c], label=f"{c} hist")
        plt.plot(
            x_plot,
            gauss(x_plot, mu, s),
            "--",
            color=colours[c],
            label=f"{c}: μ={mu:.2f}, σ={s:.2f}",
        )
        print(f"{c}: μ = {mu:.2f}, σ = {s:.2f}")

    plt.xlabel("Raw pixel value")
    plt.ylabel("Probability density")
    plt.title("Single-pixel Gaussian fit "
              f"({LOW_THRESH} < max(R,G,B) < {SAT_LIMIT})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
