#!/usr/bin/env python3
"""
record_1000_frames.py
---------------------
• Allocates a NumPy array big enough to hold 1 000 RGB frames at the webcam’s
  native resolution.
• Captures the frames in real time with OpenCV, filling the array in place.
• Prints progress every 50 frames.
• Saves the whole stack to a single compressed “.npz” archive when done.
"""

import cv2
import numpy as np
from pathlib import Path

N_FRAMES = 1000
OUT_FILE = Path("webcam_1000_frames.npz")   # change path if you like

def main() -> None:
    cap = cv2.VideoCapture(2, cv2.CAP_V4L)   # device 0; adjust if needed
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    # ---- Query native resolution ------------------------------------------------
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channels = 3  # BGR from OpenCV

    # ---- Pre-allocate big array --------------------------------------------------
    # Shape: (frames, height, width, channels), uint8 just like raw webcam output
    frames = np.empty((N_FRAMES, height, width, channels), dtype=np.uint8)

    print(f"Recording {N_FRAMES} frames at {width}×{height} …")
    for i in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Frame grab failed at index {i}.")
        frames[i] = frame                    # store in pre-allocated buffer

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Captured {i + 1}/{N_FRAMES}")

    cap.release()
    print("Capture complete. Saving…")

    # ---- Save compressed .npz ----------------------------------------------------
    np.savez_compressed(OUT_FILE, frames=frames)
    print(f"Done → {OUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
