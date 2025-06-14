#!/usr/bin/env python3
"""
robust_noise_fit.py
-------------------
Robust ML fit of

    Var[y] = σ_sensor² + α · μ

with a 2-D grid pre-search so that α ≈ 0 local minima are avoided.
"""

from __future__ import annotations
from pathlib import Path
import random, sys
import numpy as np
from scipy.special import erf
from scipy.optimize import minimize

# ---- configuration -------------------------------------------------------
PATH        = "webcam_1000_frames.npz"
N_PIXELS    = 400
T_MAX       = 1000
MU_LO, MU_HI = 10, 240          # shave off floor / clipping pixels
GRID_SZ     = 20                # grid for (σ², α)
# --------------------------------------------------------------------------

def rand_coords(h: int, w: int, n: int):
    s=set()
    while len(s)<n: s.add((random.randint(0,h-1),random.randint(0,w-1)))
    return list(s)

def bin_probs(mu, sig):
    """Return P[k] for k=0..255 given μ,σ  (vectorised)."""
    edges   = np.arange(-0.5,256.5)
    z_hi    = (edges[1:]-mu[:,None])/sig[:,None]
    z_lo    = (edges[:-1]-mu[:,None])/sig[:,None]
    c_hi    = 0.5*(1+erf(z_hi/np.sqrt(2)))
    c_lo    = 0.5*(1+erf(z_lo/np.sqrt(2)))
    p       = np.clip(c_hi-c_lo, 1e-12, None)
    return p

def neg_ll(theta, counts):
    log_sig2, log_alpha, *mu = theta
    sig2  = np.exp(log_sig2);  alpha = np.exp(log_alpha)
    mu    = np.asarray(mu)
    var   = sig2 + alpha*mu
    sig   = np.sqrt(var)
    p     = bin_probs(mu, sig)
    return -np.sum(counts*np.log(p))

def fit_channel(frames, coords, label):
    T, H, W = frames.shape
    T = min(T, T_MAX)
    ts = frames[:T, [r for r,_ in coords], [c for _,c in coords]].astype(float)
    mu_hat = ts.mean(0) ; var_hat = ts.var(0)
    ok = (mu_hat>MU_LO)&(mu_hat<MU_HI)
    ts  = ts[:, ok] ; mu_hat = mu_hat[ok] ; var_hat = var_hat[ok]
    N   = mu_hat.size
    if N < 20:
        print(f"{label}: too few usable pixels"); return

    # -- Stage A  moment-line init -------------------------------------------
    beta, *_ = np.linalg.lstsq(
        np.vstack([np.ones_like(mu_hat), mu_hat]).T, var_hat, rcond=None)

    b0, b1 = map(float, beta[:2])        # <- turn them into scalars
    b0 = max(b0, 1.0)
    b1 = max(b1, 1e-3)
    
    # -- Stage B  coarse grid search
    sig2_grid = np.geomspace(b0*0.2, b0*5, GRID_SZ)
    a_grid    = np.geomspace(b1*0.2, b1*5, GRID_SZ)
    best=(np.inf,None,None)
    counts = np.apply_along_axis(
        lambda v: np.bincount(v.astype(np.uint8), minlength=256),
        0, ts).T                        # (N,256)
    for s2 in sig2_grid:
        for a in a_grid:
            var = s2 + a*mu_hat
            ll  = -np.sum(counts*np.log(bin_probs(mu_hat,np.sqrt(var))))
            if ll<best[0]: best=(ll,s2,a)
    sig2_0, a0 = best[1], best[2]

    # -- Stage C  full L-BFGS
    theta0 = np.concatenate([[np.log(sig2_0), np.log(a0)], mu_hat])
    bounds = [(-10,10),(-10,10)] + [(1,254)]*N
    res = minimize(neg_ll, theta0, args=(counts,),
                   method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=7000,ftol=1e-8))
    if not res.success:
        print(f"{label}: optimiser failed:",res.message,file=sys.stderr);return
    log_sig2, log_alpha, *_ = res.x
    sig2   = np.exp(log_sig2);  sig_s = np.sqrt(sig2)
    alpha  = np.exp(log_alpha)
    mu_min = mu_hat.min()
    sigma_shot_min = np.sqrt(alpha*mu_min)

    print(f"\n{label} channel  ({N} px)")
    print(f"  σ_sensor         = {sig_s:8.3f} DN")
    print(f"  α (shot factor)  = {alpha:8.4f} DN²/DN")
    print(f"  μ_min            = {mu_min:8.3f} DN")
    print(f"  σ_shot,min       = {sigma_shot_min:8.3f} DN")
    print(f"  σ_total(μ)       = sqrt({sig2:.2f}+{alpha:.4f}·μ)")

def main():
    arch=Path(PATH)
    if not arch.exists(): raise FileNotFoundError(PATH)
    frames=np.load(arch)["frames"]
    T,H,W,_=frames.shape
    coords=rand_coords(H,W,N_PIXELS)
    b,g,r=(frames[...,k] for k in (0,1,2))  # BGR
    fit_channel(r,coords,"R")
    fit_channel(g,coords,"G")
    fit_channel(b,coords,"B")

if __name__=="__main__": main()
