import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
def gaussian_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-(x / sigma) ** 2 / 2.0)

def phi(z: float) -> float:
    """Standard-normal CDF Φ(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# ---------------------------------------------------------------------------
def stripe_area_ratio(sigma: float,
                      step: float = 1.0,
                      shift: float = 0.0,
                      half_range: int = 6) -> float:
    """
    Return Σ blue / Σ yellow for the Gaussian with one particular shift.
    """
    n_min = math.floor((-half_range * sigma - shift + step / 2) / step)
    n_max = math.ceil(( half_range * sigma - shift - step / 2) / step)

    sum_blue = 0.0
    sum_yellow = 0.0
    for n in range(n_min, n_max + 1):
        a = n * step + shift - step / 2
        b = n * step + shift + step / 2
        area = phi(b / sigma) - phi(a / sigma)
        if n % 2 == 0:
            sum_blue += area
        else:
            sum_yellow += area
            
    return sum_blue / sum_yellow if sum_yellow else float("nan")

# ---------------------------------------------------------------------------
def sweep_shift_and_plot_ratio(sigma: float,
                               step: float = 1.0,
                               half_range: int = 6,
                               d_shift: float = 0.01):
    """
    Compute Σ blue / Σ yellow for shifts 0, 0.01, 0.02, …, 1.00
    and plot the curve.
    """
    shifts = np.arange(0.0, 1.0 + 1e-9, d_shift)
    ratios = [stripe_area_ratio(sigma, step, s, half_range) for s in shifts]

    plt.figure(figsize=(9, 5))
    plt.plot(shifts, ratios, lw=2)
    plt.axhline(1.0, ls=":", lw=1)
    plt.xlabel("Shift (relative to stripe width)")
    plt.ylabel("Σ blue / Σ yellow")
    plt.title(f"Blue-to-Yellow Area Ratio  (σ={sigma}, step={step})")
    plt.grid(True, ls=":")
    plt.show()

# ---------------------------------------------------------------------------
# Example usage – keep your original detailed stripe plot if you still want it
# and then call the new sweep-and-plot helper:

# (1) Your existing visual of one particular shift …
# plot_gaussian_with_stripes(sigma=2.1, step=1.0, half_range=30, shift=0.30)

# (2) NEW: ratio-vs-shift curve
sweep_shift_and_plot_ratio(sigma=0.18, step=1.0, half_range=30, d_shift=0.01)
