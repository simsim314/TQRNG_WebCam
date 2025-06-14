from __future__ import annotations

"""
pixel_shifted_gauss_sigma.py (enhanced)
---------------------------------------
Record 1000 random pixels for 10 s, centre each pixel’s samples, pool by
colour channel, **FIT** N(0,σ²) to each colour’s distribution, and plot.
Additionally **print the mean (μ) pixel value for each channel** before
centering, alongside the fitted σ.
Run:
    python pixel_shifted_gauss_sigma.py
"""

import random
import time
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
NUM_PIXELS = 1000
RECORD_SEC = 10
BIN_MIN, BIN_MAX = -256, 256           # histogram support
# ---------------------------------------------------------------------------


def choose_random_coords(h: int, w: int, n: int) -> List[Tuple[int, int]]:
    coords = set()
    while len(coords) < n:
        coords.add((random.randint(0, h - 1), random.randint(0, w - 1)))
    return sorted(coords)


# ---------- Gaussian helpers ------------------------------------------------
def gauss_zero(x: np.ndarray, sigma: float) -> np.ndarray:
    """N(0,σ²) PDF evaluated at x."""
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)


def fit_sigma(x_bins: np.ndarray, pdf_emp: np.ndarray, sigma0: float) -> float:
    """
    Fit σ by least-squares against the empirical PDF in `pdf_emp`.
    Falls back to a coarse grid-search if SciPy is unavailable.
    """
    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            f=gauss_zero,
            xdata=x_bins,
            ydata=pdf_emp,
            p0=[sigma0],
            bounds=(1e-3, 500.0),
            maxfev=10000,
        )
        return float(popt[0])
    except Exception:  # SciPy absent or fit failed → tiny one-dim grid-search
        sigmas = np.linspace(max(1.0, 0.5 * sigma0), 2.0 * sigma0, num=200)
        errs = [
            np.mean((gauss_zero(x_bins, s) - pdf_emp) ** 2, dtype=float) for s in sigmas
        ]
        return float(sigmas[int(np.argmin(errs))])


# ---------- main ------------------------------------------------------------
def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam 0")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed initial frame read")
    h, w = frame.shape[:2]

    coords = choose_random_coords(h, w, NUM_PIXELS)
    print(f"Recording {NUM_PIXELS} pixels for {RECORD_SEC}s …")

    data_r, data_g, data_b = defaultdict(list), defaultdict(list), defaultdict(list)

    end_t = time.time() + RECORD_SEC
    while time.time() < end_t:
        ok, frame = cap.read()
        if not ok:
            continue
        for y, x in coords:
            b, g, r = frame[y, x]
            data_r[(y, x)].append(int(r))
            data_g[(y, x)].append(int(g))
            data_b[(y, x)].append(int(b))

    cap.release()

    # -----------------------------------------------------------------------
    # STACK *ALL* samples (before centring) to compute μ per channel
    all_r = np.concatenate([np.asarray(v, dtype=float) for v in data_r.values()])
    all_g = np.concatenate([np.asarray(v, dtype=float) for v in data_g.values()])
    all_b = np.concatenate([np.asarray(v, dtype=float) for v in data_b.values()])

    mean_r, mean_g, mean_b = all_r.mean(), all_g.mean(), all_b.mean()

    print("\nMean pixel values (raw):")
    print(f"  μ_R = {mean_r:.2f}")
    print(f"  μ_G = {mean_g:.2f}")
    print(f"  μ_B = {mean_b:.2f}")

    # -----------------------------------------------------------------------
    # Build centred sample arrays for σ fitting
    centred_r, centred_g, centred_b = [], [], []
    for coord in coords:
        for src, dest in (
            (data_r[coord], centred_r),
            (data_g[coord], centred_g),
            (data_b[coord], centred_b),
        ):
            arr = np.asarray(src, dtype=float)
            dest.extend(arr - arr.mean())

    centred_r, centred_g, centred_b = map(np.asarray, (centred_r, centred_g, centred_b))

    # -----------------------------------------------------------------------
    # Histogram (empirical PDF) for fitting
    bins = np.arange(BIN_MIN - 0.5, BIN_MAX + 1.5)
    x_centres = np.arange(BIN_MIN, BIN_MAX + 1)

    hr, _ = np.histogram(centred_r, bins=bins, density=True)
    hg, _ = np.histogram(centred_g, bins=bins, density=True)
    hb, _ = np.histogram(centred_b, bins=bins, density=True)

    # Initial guesses: use sample SD
    sigma_r0, sigma_g0, sigma_b0 = centred_r.std(ddof=0), centred_g.std(ddof=0), centred_b.std(ddof=0)

    # ***** FIT σ *****
    sigma_r = fit_sigma(x_centres, hr, sigma_r0)
    sigma_g = fit_sigma(x_centres, hg, sigma_g0)
    sigma_b = fit_sigma(x_centres, hb, sigma_b0)

    print("\nGaussian σ estimates via histogram fit:")
    print(f"  σ_R = {sigma_r:.2f}")
    print(f"  σ_G = {sigma_g:.2f}")
    print(f"  σ_B = {sigma_b:.2f}")

    # -----------------------------------------------------------------------
    # Plot
    plt.figure("Centred data & Gaussian fit (1000 px, 10 s)")
    plt.step(x_centres, hr, where="mid", color="red",   label="ΔR hist")
    plt.step(x_centres, hg, where="mid", color="green", label="ΔG hist")
    plt.step(x_centres, hb, where="mid", color="blue",  label="ΔB hist")

    plt.plot(x_centres, gauss_zero(x_centres, sigma_r), "--", color="red",   label=f"N(0,{sigma_r:.1f}²)")
    plt.plot(x_centres, gauss_zero(x_centres, sigma_g), "--", color="green", label=f"N(0,{sigma_g:.1f}²)")
    plt.plot(x_centres, gauss_zero(x_centres, sigma_b), "--", color="blue",  label=f"N(0,{sigma_b:.1f}²)")

    plt.xlabel("Value − per-pixel mean")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
