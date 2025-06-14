import math
import numpy as np
import matplotlib.pyplot as plt

def gaussian_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-(x / sigma) ** 2 / 2.0)

def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def plot_gaussian_with_stripes(
    sigma: float,
    step: float = 1.0,
    shift: float = 0.0,
    half_range: int = 6,
    threshold: float = 1e-4,          # skip stripes & labels when P < threshold
):
    """
    Gaussian with alternating stripe areas.
       • blue  = even‑index stripes, yellow = odd
       • Stripe skipped entirely when its probability < *threshold*.
       • Area labels placed just above the x‑axis.
       • Title shows Σ blue and Σ yellow with 6 significant digits.
    """
    # ------ precompute ----------------------------------------------------
    x_min = -half_range * sigma
    x_max =  half_range * sigma
    x     = np.linspace(x_min, x_max, 4000)
    pdf   = gaussian_pdf(x, sigma)
    max_pdf = pdf.max()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, pdf, color="black", lw=2, label="Gaussian PDF")

    # stripe index range that can intersect the visible window
    n_min = math.floor((x_min - shift + step / 2) / step)
    n_max = math.ceil((x_max - shift - step / 2) / step)

    sum_blue   = 0.0
    sum_yellow = 0.0

    for n in range(n_min, n_max + 1):
        a = n * step + shift - step / 2
        b = n * step + shift + step / 2
        area = phi(b / sigma) - phi(a / sigma)

        colour = "blue" if n % 2 == 0 else "yellow"
        if n % 2 == 0:
            sum_blue += area
        else:
            sum_yellow += area

        if area < threshold:
            continue  # treat as zero → skip drawing & labelling

        # visible part (clip inside window)
        a_vis = max(a, x_min)
        b_vis = min(b, x_max)
        if a_vis < b_vis:
            xs   = np.linspace(a_vis, b_vis, 200)
            ax.fill_between(xs, gaussian_pdf(xs, sigma),
                            color=colour, alpha=0.4, edgecolor="none")

        # area label near x‑axis
        mid_x = (a + b) / 2
        if x_min <= mid_x <= x_max:
            ax.text(
                mid_x,
                0.02 * max_pdf,   # just above baseline
                f"{area:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

    title = (f"Gaussian σ={sigma}, step={step}, shift={shift}  "
             f"(Σ blue={sum_blue:.9g}, Σ yellow={sum_yellow:.9g})")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Probability density")
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.show()

# -------------------------------------------------------------------------
plot_gaussian_with_stripes(sigma=0.33, step=1.0, half_range = 30, shift=0.0)
