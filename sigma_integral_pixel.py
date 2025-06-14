
import math
from typing import Tuple


# ---------------------------------------------------------------------------
def _phi(z: float) -> float:
    """Standard-normal CDF Φ(z) = ½[1 + erf(z/√2)]."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def stripe_areas(
    sigma: float,
    half_range: int = 10,
    *,
    verbose: bool = True,
    shift: float = 0
):
    """
    Integrate N(0,σ²) over 1-unit stripes shifted by 0.5, and print
    cumulative (“partial”) sums:
      white :  n = 0, ±2, ±4, …
      black :  n = ±1, ±3, ±5, …

    Returns
    -------
    white_prob, black_prob, tail_prob
    """
    import math

    # ---------- helpers --------------------------------------------------
    def _phi(z: float) -> float:                # Φ(z)  CDF of N(0,1)
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _stripe_prob(n: int, shift: float) -> float:         # P(X in stripe n)
        a, b = n - 0.5 + shift, n + 0.5 + shift
        return _phi(b / sigma) - _phi(a / sigma)

    # ---------- stripe index range fully inside [-half_rangeσ,+half_rangeσ)
    n_min = math.ceil(-half_range * sigma + 0.5)
    n_max = math.floor( half_range * sigma - 0.5)
    max_k  = max(-n_min, n_max)               # |n| extent

    # ---------- probability for every stripe -----------------------------
    stripe_p = {n: _stripe_prob(n, shift) for n in range(n_min, n_max + 1)}

    # ---------- cumulative sums ------------------------------------------
    white_cum = 0.0
    black_cum = 0.0
    white_partials = []        # list of (max |n|, cumulative P_white)
    black_partials = []        # list of (max |n|, cumulative P_black)

    # --- white stripes: n = 0, ±2, ±4, ...
    if 0 in stripe_p:                          # centre stripe
        white_cum += stripe_p[0]
        white_partials.append((0, white_cum))

    for k in range(2, max_k + 1, 2):
        p_k = stripe_p.get(k, 0.0) + stripe_p.get(-k, 0.0)
        if p_k == 0.0:
            break
        white_cum += p_k
        white_partials.append((k, white_cum))

    # --- black stripes: n = ±1, ±3, ±5, ...
    for k in range(1, max_k + 1, 2):
        p_k = stripe_p.get(k, 0.0) + stripe_p.get(-k, 0.0)
        if p_k == 0.0:
            break
        black_cum += p_k
        black_partials.append((k, black_cum))

    # ---------- grand totals and tail ------------------------------------
    white_prob = white_cum
    black_prob = black_cum
    tail_prob  = 1.0 - (white_prob + black_prob)

    # ---------- printing --------------------------------------------------
    if verbose:
        print(f"\nσ = {sigma:.3f}   stripes fully inside n = {n_min} … {n_max}")
        print("White (even-n) partial sums:")
        for k, p in white_partials:
            label = "0" if k == 0 else f"0, ±2 … ±{k}"
            print(f"   up to {label:<8}:  {p:.10g}")
        print("Black (odd-n)  partial sums:")
        for k, p in black_partials:
            label = f"±1 … ±{k}"
            print(f"   up to {label:<8}:  {p:.10g}")
        print("-------------------------------------------------")
        print(f" total white : {white_prob:.10g}")
        print(f" total black : {black_prob:.10g}")
        print(f" outside ±{half_range}σ : {tail_prob:.10g}\n")

    return white_prob, black_prob, tail_prob


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SIGMAS = [0.5]
    for s in SIGMAS:
        white, black, _ = stripe_areas(s, half_range=30, shift = 0.4)
        print(f"σ = {s:4.2f}  white = {white:.8f}, black = {black:.8f}")
