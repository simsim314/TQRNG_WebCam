#!/usr/bin/env python3
"""
cam_controller.py  –  Manual shutter for a UVC webcam (no GUI).

• On launch:
      – Finds the YUYV mode with highest FPS (ties → largest resolution)
      – Switches the driver to that mode
      – Locks out *all* automatics (WB, dynamic framerate, etc.)
      – Sets:
            – exposure_absolute = 50
            – gamma = 100
      – Starts a live preview

NOTE: This webcam exposes no hardware gain (V4L2_CID_GAIN is missing),
so we use only physical exposure.
"""

import subprocess, cv2, re
import numpy as np 


cam_idx = 2
DEVICE = f"/dev/video{cam_idx}"
cap = None
cam_w, cam_h = 640, 480

EXPOSURE_VAL = 1       # in 100 µs units → 5 ms
#GAMMA_VAL    = 500      # driver-dependent, often [72..500]
CONTRAST_VAL = 50

# ── Helper to set v4l2 control ─────────────────────────────────────────────
def v4l2_set(control, value):
    subprocess.run(["v4l2-ctl", "-d", DEVICE, "-c", f"{control}={value}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── Find best YUYV mode ────────────────────────────────────────────────────
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
        raise RuntimeError("device advertises no YUYV modes")
    return max(modes, key=lambda m: (m[2], m[0]*m[1]))

# ── Initial setup ──────────────────────────────────────────────────────────
def init_camera():
    global cam_w, cam_h
    w, h, fps = best_yuyv_mode()
    cam_w, cam_h = w, h
    print(f"Selected YUYV {w}×{h} @ {fps:.1f} fps")

    subprocess.run(
        ["v4l2-ctl", "-d", DEVICE,
         f"--set-fmt-video=width={w},height={h},pixelformat=YUYV",
         f"--set-parm={int(round(1/fps*1000))}"],
        check=True)

    # Disable all automatics
    v4l2_set("white_balance_automatic",     0)
    v4l2_set("exposure_auto",               1)  # 1 = manual mode
    v4l2_set("exposure_auto_priority",      0)
    v4l2_set("exposure_dynamic_framerate",  0)
    v4l2_set("backlight_compensation",      0)

    # Apply manual settings
    v4l2_set("exposure_time_absolute", EXPOSURE_VAL)
    v4l2_set("contrast", CONTRAST_VAL)
    #v4l2_set("gamma", GAMMA_VAL)

# ── Preview ────────────────────────────────────────────────────────────────
def preview():
    global cap
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # keep raw YUYV
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # Flatten raw YUYV frame and extract every even byte (Y channel)
        raw = frame.reshape(-1)        # 1D array of bytes
        y    = raw[::2]                # Y values at every even index

        # Reshape into grayscale image
        y_image = y.reshape((cam_h, cam_w)).astype(np.uint8)
        
        print(np.min(y_image), np.max(y_image))
        unique_vals = np.unique(y_image)
        print(f"Y range: {y_image.min()}–{y_image.max()} | Unique values ({len(unique_vals)}):", unique_vals)
        cv2.imshow("Y channel (raw)", y_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_camera()
    preview()
