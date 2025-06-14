#!/usr/bin/env python3
"""
play_1000_frames.py
-------------------
Load `webcam_1000_frames.npz` and play it back frame-by-frame with OpenCV.

Controls
--------
• Esc or q   – quit early
"""

import cv2
import numpy as np
from pathlib import Path
import time

ARCHIVE = Path("webcam_1000_frames.npz")
FPS = 30                     # target playback speed
DELAY = int(1000 / FPS)      # cv2.waitKey expects milliseconds

def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)

    print(f"Loading {ARCHIVE} …")
    frames = np.load(ARCHIVE)["frames"]       # shape = (N, H, W, 3), dtype=uint8
    n_frames = len(frames)
    print(f"{n_frames} frames loaded, starting playback …")

    cv2.namedWindow("Playback", cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)

    start = time.time()
    for i, frame in enumerate(frames):
        cv2.imshow("Playback", frame)
        key = cv2.waitKey(DELAY) & 0xFF
        if key in (ord("q"), 27):             # 27 = Esc
            break
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Shown {i + 1}/{n_frames}")

    cv2.destroyAllWindows()
    elapsed = time.time() - start
    print(f"Finished ({elapsed:.2f} s).")

if __name__ == "__main__":
    main()
