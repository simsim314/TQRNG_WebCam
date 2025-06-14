#setup camera - remove anything auto gain nor WB most linear to sensor data with minimal distortion
#set maximum resolution 
#set maximum fps 
#recording of 1000 frames
#analyze the recording on 10K pixles find:
#minimal sigma fit per channel
#use blue-yellow stripe script to compute worst case ratio bias blue/yellow for those sigma 
#compute bias of single worst case scenario pixel
#record this number - this is callibration number 
#reoprt max pixel count etc. 
#setup class with bias to be 1e-06 and either using calibration or using value 0.33
#class can be adjusted to be less biased or more biased (for simple cases 1e-3 for scientific 1e-9)
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
max_webcam_yuyv.py
------------------
Scan webcams, disable auto settings, and find highest YUYV resolution × FPS (pixel/sec).

Michael Simkin © 2025-06-13
"""

import subprocess
import re
import os
import cv2

RAW_CODES = {"YUYV", "YUY2", "UYVY"}  # Uncompressed YUV formats

def scan_available_cameras(max_index=9):
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap is None or not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        cap.release()
        if ok:
            available.append(idx)
    return available

def disable_auto_controls(dev_index):
    device = f"/dev/video{dev_index}"
    controls = [
        "white_balance_temperature_auto=0",
        "exposure_auto=1",     # Manual mode
        "gain_auto=0"
    ]
    for ctrl in controls:
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", device, "-c", ctrl],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        except Exception:
            pass

def list_formats_and_fps(dev_index: int):
    device = f"/dev/video{dev_index}"
    try:
        raw = subprocess.check_output(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []

    formats = []
    pixfmt, w, h = None, None, None

    for line in raw.splitlines():
        m = re.match(r"\s*\[\d+\]:\s*'([^']+)'", line)
        if not m:
            m = re.match(r"\s*Pixel\s+Format:\s*'([^']+)'", line)
        if m:
            pixfmt = m.group(1)
            continue

        m = re.search(r"(\d+)x(\d+)", line)
        if m and "Size:" in line:
            w, h = map(int, m.groups())
            continue

        if "Interval:" in line:
            m_ratio = re.search(r"(\d+)/(\d+)", line)
            if m_ratio:
                num, den = map(int, m_ratio.groups())
                fps = den / num if num else 0.0
            else:
                m_fps = re.search(r"\(([\d.]+)\s*fps", line)
                fps = float(m_fps.group(1)) if m_fps else 0.0

            if pixfmt and w and h and fps:
                formats.append(
                    dict(width=w,
                         height=h,
                         fps=fps,
                         pixfmt=pixfmt,
                         px_per_sec=w * h * fps)
                )

    return formats

def main(max_index=9):
    print(f"🔍 Scanning webcams 0–{max_index} ...")
    available = scan_available_cameras(max_index)

    if not available:
        print("❌ No cameras available.")
        return

    for idx in available:
        print(f"\n🎥 /dev/video{idx}")
        disable_auto_controls(idx)
        formats = list_formats_and_fps(idx)

        yuyv_modes = [f for f in formats if f["pixfmt"] in RAW_CODES]
        if not yuyv_modes:
            print(f"  ⚠️  No YUYV/YUY2/UYVY modes found.")
            continue

        best = max(yuyv_modes, key=lambda f: f["px_per_sec"])
        print(f"  ✅ Best {best['pixfmt']}: {best['width']}x{best['height']} @ {best['fps']:.2f} FPS")
        print(f"     Pixels/sec: {int(best['px_per_sec']):,}")

if __name__ == "__main__":
    import sys
    max_index = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    main(max_index)
