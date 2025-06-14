"""quantum_cam_demo.py
================================
Generate 10 000 quantum‑random floats using a larger 100‑pixel patch size
and plot their distribution in 50 histogram bins.

Run with
    python quantum_cam_demo.py
"""
import matplotlib
matplotlib.use("TkAgg")  

import matplotlib.pyplot as plt

import numpy as np

# Import the RNG library (assumes quantum_cam_rng.py is in the same dir or PYTHONPATH)
from quantum_cam_rng import QuantumCamRNG as random

# --- Configure RNG -----------------------------------------------------------
# Change the default patch size to 5 pixels. Doing this once affects the
# subsequent static method calls.
random.init(patch_size=100, webcam_index = 2)

# --- Generate data -----------------------------------------------------------
NUM_SAMPLES = 200
samples = random.random(NUM_SAMPLES)
random.close()  # release webcam ASAP

# --- Plot histogram ----------------------------------------------------------
plt.figure("Quantum‑random floats (N=10k)")
plt.hist(samples, bins=10, edgecolor="black")
plt.title("Histogram of 10 000 quantum‑random floats")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
