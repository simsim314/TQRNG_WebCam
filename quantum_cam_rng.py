"""quantum_cam_rng.py
=====================================
A webcam‑based true‑random‑number generator that mirrors much of the
``random`` standard‑library API while always returning **NumPy arrays**.

Highlights (v0.9)
-----------------
* **NEW**: Discards pixel values outside the usable intensity window
  ``15 ≤ val ≤ 240`` before extracting their least‑significant bits.
* Flattened RGB‑interleaved bitstream (R,G,B, R,G,B, …).
* ``patch_size`` counts **pixels** (e.g., 100 → 300 bits/patch).
* XOR‑reduced (default) or ``full_patch_bits=True`` to keep every bit.
* Helpers: ``randint``, ``randrange``, ``random``, ``choice``,
  ``choices``, ``shuffle``, ``sample``.
* Demo at the bottom exercises every helper.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple, Union, List

import cv2
import numpy as _np

# ---------------------------------------------------------------------------
# Optional GPU backend
# ---------------------------------------------------------------------------
try:
    import cupy as _cupy  # type: ignore

    _xp = _cupy  # type: ignore
    _USING_GPU = True
except ImportError:  # CPU fallback
    _xp = _np  # type: ignore
    _USING_GPU = False

# ---------------------------------------------------------------------------
# Backend → NumPy helper
# ---------------------------------------------------------------------------

def _as_numpy(arr):
    """Return *arr* as NumPy, copying from GPU if necessary."""
    if _USING_GPU and isinstance(arr, _xp.ndarray):  # type: ignore
        return _xp.asnumpy(arr)  # type: ignore[attr-defined]
    return _np.asarray(arr)

# ---------------------------------------------------------------------------
# GPU XOR helper (CuPy lacks reduce for xor)
# ---------------------------------------------------------------------------

def _xor_reduce_gpu(arr: "_cupy.ndarray") -> "_cupy.ndarray":  # noqa: F821
    res = arr[:, 0].copy()
    for i in range(1, arr.shape[1]):
        res ^= arr[:, i]
    return res

@lru_cache(maxsize=1)
def _powers_of_two(dtype=_xp.uint64):
    return _xp.power(_xp.array(2, dtype=dtype), _xp.arange(63, -1, -1, dtype=dtype))

# ---------------------------------------------------------------------------
# RNG class
# ---------------------------------------------------------------------------

class QuantumCamRNG:
    """Static‑style RNG backed by a webcam sensor."""

    # Configurable class‑wide settings
    _patch_size: int = 50         # pixels per patch
    _webcam_index: int = 0
    _full_patch_bits: bool = False

    # Internal camera singleton
    _camera: "QuantumCamRNG._Camera | None" = None

    # Expose backend details
    xp = _xp
    _using_gpu = _USING_GPU

    # ------------------------------------------------------------------
    # Constructor to update class defaults
    # ------------------------------------------------------------------
    @classmethod
    def init(cls, *, patch_size: int | None = None, webcam_index: int | None = None, full_patch_bits: bool | None = None):
        if patch_size is not None:
            cls._patch_size = int(patch_size)
        if full_patch_bits is not None:
            cls._full_patch_bits = bool(full_patch_bits)
        if webcam_index is not None and webcam_index != cls._webcam_index:
            # Switch camera → close old one first
            cls.close()
            cls._webcam_index = int(webcam_index)

    # ------------------------------------------------------------------
    # Camera wrapper
    # ------------------------------------------------------------------
    class _Camera:
        """Wrapper that keeps a re‑usable interleave buffer."""

        # Intensity window constants
        _MIN_VAL = 15
        _MAX_VAL = 240

        def __init__(self, index: int):
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open webcam (index {index})")
            self._buf = None  # lazily allocated interleave buffer

        def close(self):
            if self.cap.isOpened():
                self.cap.release()

        def frame_bits(self, cls: "type[QuantumCamRNG]") -> "_xp.ndarray":  # noqa: F821
            """Read one frame and return an interleaved 1‑bit RGB stream.

            Pixels with any channel outside the [15, 240] range are *ignored*.
            The three colour channels are kept in lock‑step; if the usable
            counts differ the shortest channel length determines the batch.
            """
            ok, frame_bgr = self.cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from webcam")

            frame = cls.xp.asarray(frame_bgr, dtype=cls.xp.uint8)

            # Split channels & create intensity masks (per channel)
            r_chan = frame[:, :, 2]
            g_chan = frame[:, :, 1]
            b_chan = frame[:, :, 0]

            r_mask = (r_chan >= self._MIN_VAL) & (r_chan <= self._MAX_VAL)
            g_mask = (g_chan >= self._MIN_VAL) & (g_chan <= self._MAX_VAL)
            b_mask = (b_chan >= self._MIN_VAL) & (b_chan <= self._MAX_VAL)

            # Extract LSBs as uint8 vectors
            r_bits = (r_chan & 1).ravel()
            g_bits = (g_chan & 1).ravel()
            b_bits = (b_chan & 1).ravel()

            # Apply masks & keep equal count across channels
            r_filtered = r_bits[r_mask.ravel()]
            g_filtered = g_bits[g_mask.ravel()]
            b_filtered = b_bits[b_mask.ravel()]

            n_valid = int(cls.xp.minimum(cls.xp.minimum(r_filtered.size, g_filtered.size), b_filtered.size))
            if n_valid == 0:
                # No sufficiently exposed pixels this frame
                return cls.xp.empty(0, dtype=cls.xp.uint8)

            r_filtered = r_filtered[:n_valid]
            g_filtered = g_filtered[:n_valid]
            b_filtered = b_filtered[:n_valid]

            # Prepare / reuse interleave buffer
            required = n_valid * 3
            if self._buf is None or self._buf.size != required:
                self._buf = cls.xp.empty(required, dtype=cls.xp.uint8)
            inter = self._buf
            inter[0::3], inter[1::3], inter[2::3] = r_filtered, g_filtered, b_filtered
            return inter

    # ------------------------------------------------------------------
    # Helper getters / processors
    # ------------------------------------------------------------------
    @classmethod
    def _cam(cls):
        if cls._camera is None:
            cls._camera = cls._Camera(cls._webcam_index)
        return cls._camera

    @classmethod
    def _patch_process(cls, flat_bits: "_xp.ndarray") -> "_xp.ndarray":  # noqa: F821
        bits_per_patch = cls._patch_size * 3
        if cls._full_patch_bits:
            return flat_bits
        n = flat_bits.size // bits_per_patch
        if n == 0:
            return cls.xp.empty(0, dtype=cls.xp.uint8)
        trimmed = flat_bits[: n * bits_per_patch].reshape(n, bits_per_patch)
        if cls._using_gpu:
            return _xor_reduce_gpu(trimmed)  # type: ignore[arg-type]
        return _np.bitwise_xor.reduce(trimmed, axis=1)  # type: ignore[arg-type]

    @classmethod
    def _bits_to_uint64(cls, bits: "_xp.ndarray") -> "_xp.ndarray":  # noqa: F821
        n = bits.size // 64
        if n == 0:
            return cls.xp.empty(0, dtype=cls.xp.uint64)
        shaped = bits[: n * 64].reshape(n, 64)
        return cls.xp.dot(shaped.astype(cls.xp.uint64), _powers_of_two())

    @staticmethod
    def _u64_to_float32(u64: "_xp.ndarray") -> _np.ndarray:  # noqa: F821
        max_u64 = _np.iinfo(_np.uint64).max
        return (_as_numpy(u64).astype(_np.float32) / max_u64).astype(_np.float32)

    # ------------------------------------------------------------------
    # Bit / float collectors (always NumPy)
    # ------------------------------------------------------------------
    @classmethod
    def _collect_bits(cls, n: int) -> _np.ndarray:
        out: List[_np.ndarray] = []
        total = 0
        while total < n:
            bits = cls._patch_process(cls._cam().frame_bits(cls))
            if bits.size == 0:
                continue  # skip empty frames
            arr = _as_numpy(bits).astype(_np.uint8, copy=False)
            out.append(arr)
            total += arr.size
        return _np.concatenate(out)[:n]

    @classmethod
    def _collect_floats(cls, n: int) -> _np.ndarray:
        out: List[_np.ndarray] = []
        total = 0
        while total < n:
            bits = cls._patch_process(cls._cam().frame_bits(cls))
            if bits.size == 0:
                continue
            out.append(cls._u64_to_float32(cls._bits_to_uint64(bits)))
            total += out[-1].size
        return _np.concatenate(out)[:n].astype(_np.float32, copy=False)

    # ------------------------------------------------------------------
    # Public numeric API
    # ------------------------------------------------------------------
    @classmethod
    def random(cls, size: Union[int, Sequence[int], Tuple[int, ...]] = 1) -> _np.ndarray:
        if isinstance(size, int):
            need, shape = size, (size,)
        else:
            shape = tuple(size)  # type: ignore[arg-type]
            need = int(_np.prod(shape))
        return cls._collect_floats(need).reshape(shape)

    @classmethod
    def randint(cls, low: int, high: int | None = None, size: Union[int, Sequence[int], Tuple[int, ...]] = 1, dtype=_np.int32,) -> _np.ndarray | int:
        if high is None:
            low, high = 0, low
        if high <= low:
            raise ValueError("high must be > low")
        if isinstance(size, int):
            need, shape = size, (size,)
        else:
            shape = tuple(size)  # type: ignore[arg-type]
            need = int(_np.prod(shape))
        floats = cls._collect_floats(need)
        ints = ((floats * (high - low)).astype(dtype) + low).reshape(shape)
        return ints if need != 1 else ints.item()

    @classmethod
    def randrange(cls, start: int, stop: int | None = None, step: int = 1) -> int:
        if step == 0:
            raise ValueError("step must not be zero")
        if stop is None:
            start, stop = 0, start
        n = (stop - start + (step - 1)) // step
        if n <= 0:
            raise ValueError("empty range")
        k = int(cls.randint(0, n))
        return start + k * step
    
    @classmethod
    def choice(cls, seq: Sequence):
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        idx = int(cls.randint(0, len(seq)))
        return seq[idx]

    @classmethod
    def choices(cls, seq: Sequence, k: int = 1) -> List:
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        idxs = cls.randint(0, len(seq), size=k).tolist()
        return [seq[i] for i in idxs]

    @classmethod
    def shuffle(cls, seq: list) -> None:
        n = len(seq)
        for i in range(n - 1, 0, -1):
            j = int(cls.randint(0, i + 1))
            seq[i], seq[j] = seq[j], seq[i]

    @classmethod
    def sample(cls, seq: Sequence, k: int) -> List:
        if k < 0 or k > len(seq):
            raise ValueError("Sample larger than population or negative")
        idxs = list(range(len(seq)))
        cls.shuffle(idxs)  # in‑place Fisher–Yates on the indices
        return [seq[i] for i in idxs[:k]]

    # ------------------------------------------------------------------
    # Clean‑up
    # ------------------------------------------------------------------
    @classmethod
    def close(cls):
        if cls._camera is not None:
            cls._camera.close()
            cls._camera = None


RNG = QuantumCamRNG

__all__ = ["QuantumCamRNG", "RNG", "main"]

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    """Run a rich demonstration of the RNG helpers (Ctrl+C to quit)."""
    print("Quantum‑camera RNG demo – Ctrl+C to quit")
    try:
        # Configure: 100‑pixel patches (300 bits) in XOR mode initially
        RNG.init(patch_size=10)

        # --- Frame statistics ------------------------------------------------
        cam = RNG._cam()
        raw_bits = cam.frame_bits(RNG)
        xor_bits = RNG._patch_process(raw_bits)
        print("Raw interleaved bits in frame:", len(raw_bits))
        print("XOR‑mode bits in frame:", xor_bits.size)

        # Switch to full‑patch harvesting and show new bit count
        RNG.init(full_patch_bits=True)
        full_bits = RNG._patch_process(raw_bits)
        print("Full‑patch bits in frame:", full_bits.size)

        # --- Numeric helper showcase ----------------------------------------
        print("Helper function examples:")
        print("randint(1, 100):", int(RNG.randint(1, 101)))  # inclusive high bound
        print("randrange(1, 100, 5):", RNG.randrange(1, 100, 5))

        seq = [10, 20, 30, 40, 50]
        print("choice(seq):", RNG.choice(seq))
        print("choices(seq, k=3):", RNG.choices(seq, k=3))
        print("sample(seq, k=2):", RNG.sample(seq, k=2))

        to_shuffle = [1, 2, 3, 4, 5]
        RNG.shuffle(to_shuffle)
        print("shuffle([1,2,3,4,5]):", to_shuffle)

    finally:
        RNG.close()
        print("Camera closed.")


if __name__ == "__main__":
    main()
