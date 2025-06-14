#!/usr/bin/env python3
"""
record_1000_y_frames.py
-----------------------
• Applies manual camera controls via V4L2
• Captures 1000 Y-channel (luminance) frames in raw YUYV format
• Saves them as a compressed .npz archive for later analysis
"""

import subprocess, cv2, re
import numpy as np
from pathlib import Path

# ── Camera config ─────────────────────────────────────────────────────────
cam_idx     = 2
DEVICE      = f"/dev/video{cam_idx}"
EXPOSURE    = 1       # ×100 µs
CONTRAST    = 50
N_FRAMES    = 1000
OUT_FILE    = Path("webcam_1000_y_frames.npz")

# ── Helper: v4l2 manual setting ───────────────────────────────────────────
def v4l2_set(control, value):
    subprocess.run(["v4l2-ctl", "-d", DEVICE, "-c", f"{control}={value}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── Find best YUYV mode ───────────────────────────────────────────────────
def best_yuyv_mode():
    txt = subprocess.check_output(["v4l2-ctl", "-d", DEVICE, "--list-formats-ext"], text=True)
    modes, fmt, size = [], None, None
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("["):
            fmt = "YUYV" if "'YUYV'" in line else None
        elif fmt == "YUYV" and line.startswith("Size:"):
            m = re.search(r"(\d+)x(\d+)", line)
            if m: size = (int(m[1]), int(m[2]))
        elif fmt == "YUYV" and size and line.startswith("Interval:"):
            m = re.search(r"Interval: Discrete ([\d.]+)s", line)
            if m:
                fps = 1.0 / float(m[1])
                modes.append((*size, fps))
    if not modes:
        raise RuntimeError("No YUYV modes found.")
    return max(modes, key=lambda m: (m[2], m[0]*m[1]))  # prefer FPS, then resolution

# ── Apply manual camera setup ─────────────────────────────────────────────
def init_camera():
    global cam_w, cam_h
    w, h, fps = best_yuyv_mode()
    cam_w, cam_h = w, h
    print(f"Selected YUYV {w}×{h} @ {fps:.1f} fps")

    subprocess.run(["v4l2-ctl", "-d", DEVICE,
                    f"--set-fmt-video=width={w},height={h},pixelformat=YUYV",
                    f"--set-parm={int(round(1/fps*1000))}"],
                   check=True)

    v4l2_set("white_balance_automatic",     0)
    v4l2_set("exposure_auto",               1)
    v4l2_set("exposure_auto_priority",      0)
    v4l2_set("exposure_dynamic_framerate",  0)
    v4l2_set("backlight_compensation",      0)
    v4l2_set("exposure_time_absolute",      EXPOSURE)
    v4l2_set("contrast",                    CONTRAST)

# ── Record Y frames only ──────────────────────────────────────────────────
def record_y_frames():
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    frames = np.empty((N_FRAMES, cam_h, cam_w), dtype=np.uint8)

    print(f"Recording {N_FRAMES} Y-frames at {cam_w}×{cam_h}…")
    for i in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Frame grab failed at index {i}")
        raw = frame.reshape(-1)
        y = raw[::2].reshape((cam_h, cam_w)).astype(np.uint8)
        frames[i] = y

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Captured {i + 1}/{N_FRAMES}")

    cap.release()
    print("Capture complete. Saving…")
    np.savez_compressed(OUT_FILE, y_frames=frames)
    print(f"Done → {OUT_FILE.resolve()}")

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_camera()
    record_y_frames()
