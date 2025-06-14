#!/usr/bin/env python3
"""
scan_webcams.py
---------------                             Michael Simkin © 2025-06-12

Quickly scan likely device indices and list the ones that open
successfully with OpenCV.

Usage
-----
    python scan_webcams.py          # probes indices 0-9
    python scan_webcams.py 15       # probes indices 0-14
"""

import sys
import cv2

def scan(max_index: int = 9) -> None:
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)  # use CAP_DSHOW on Windows
        if cap is None or not cap.isOpened():
            cap.release()
            continue
        # Optional: try grabbing a frame to be extra sure
        ok, _ = cap.read()
        cap.release()
        if ok:
            available.append(idx)

    if available:
        print("✅ Available camera indices:", *available)
    else:
        print("❌ No cameras found in range 0–", max_index)

if __name__ == "__main__":
    # Allow an optional upper-bound argument
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    scan(n)
