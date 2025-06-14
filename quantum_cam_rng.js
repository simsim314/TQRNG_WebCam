/* quantum_cam_rng.js – v0.2 (browser‑only)
   ------------------------------------------------------
   Fixes:
   • Removed signed‑shift overflow in `_bitsToFloat32` (no more negative
     floats ⇒ `randint` cannot yield negatives; shuffle/sample no longer
     produce `null`).
   • Minor doc and safety tweaks.
*/

export class QuantumCamRNG {
  /* ------------------ construction & camera ------------------ */
  constructor({ patchSize = 50, fullPatchBits = false, videoConstraints = { video: { width: 640, height: 480 } } } = {}) {
    this.patchSize = patchSize;           // pixels per patch
    this.fullPatchBits = fullPatchBits;   // keep every bit vs XOR‑reduce
    this.constraints = videoConstraints;  // passed to getUserMedia

    this._stream = null;          // MediaStream from getUserMedia
    this._video = null;           // <video> element (hidden)
    this._canvas = null;          // off‑screen <canvas>
    this._ctx = null;             // 2D context
    this._buf = null;             // Uint8Array interleave buffer

    this._MIN_VAL = 15;
    this._MAX_VAL = 240;
  }

  /* Start webcam & allocate helpers.  Returns **this** once ready. */
  async start() {
    if (this._stream) return this; // already started

    // 1) Request camera
    this._stream = await navigator.mediaDevices.getUserMedia(this.constraints);

    // 2) Hidden video sink
    this._video = document.createElement("video");
    this._video.style.display = "none";
    this._video.srcObject = this._stream;
    this._video.playsInline = true;
    await this._video.play(); // wait until metadata loaded & playback begun

    // 3) Off‑screen canvas matching video size
    this._canvas = document.createElement("canvas");
    this._canvas.width = this._video.videoWidth;
    this._canvas.height = this._video.videoHeight;
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });

    return this;
  }

  /* Stop camera & release resources. */
  stop() {
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
    this._video?.remove();
    this._video = null;
    this._canvas = null;
    this._ctx = null;
    this._buf = null;
  }

  /* ------------------ low‑level helpers ------------------ */
  /* Capture a single frame; return **Uint8Array** of interleaved bits. */
  _frameBits() {
    // Draw current video frame on canvas
    this._ctx.drawImage(this._video, 0, 0, this._canvas.width, this._canvas.height);
    const { data } = this._ctx.getImageData(0, 0, this._canvas.width, this._canvas.height);
    // data is Uint8ClampedArray in RGBA order

    // Allocate / reuse buffer for interleaved bits (RGB only)
    const nPix = data.length >>> 2; // divide by 4 using bitwise
    if (!this._buf || this._buf.length !== nPix * 3) {
      this._buf = new Uint8Array(nPix * 3);
    }

    let count = 0; // number of **usable** pixels
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      if (r < this._MIN_VAL || r > this._MAX_VAL) continue;
      if (g < this._MIN_VAL || g > this._MAX_VAL) continue;
      if (b < this._MIN_VAL || b > this._MAX_VAL) continue;

      this._buf[count * 3] = r & 1;
      this._buf[count * 3 + 1] = g & 1;
      this._buf[count * 3 + 2] = b & 1;
      count += 1;
    }
    return this._buf.subarray(0, count * 3); // trim to valid content
  }

  /* XOR‑reduce patches unless fullPatchBits=true. */
  _patchProcess(bits) {
    if (this.fullPatchBits) return bits;
    const bitsPerPatch = this.patchSize * 3;
    const n = Math.floor(bits.length / bitsPerPatch);
    if (n === 0) return new Uint8Array(0);

    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      const offset = i * bitsPerPatch;
      for (let j = 0; j < bitsPerPatch; j++) acc ^= bits[offset + j];
      out[i] = acc;
    }
    return out;
  }

  /* Collect ≥ n bits (Uint8Array) – may span multiple frames. */
  async _collectBits(n) {
    const chunks = [];
    let total = 0;
    while (total < n) {
      await new Promise(requestAnimationFrame); // yield to next frame
      const bits = this._patchProcess(this._frameBits());
      if (bits.length === 0) continue; // skip empty frame
      chunks.push(bits);
      total += bits.length;
    }
    const out = new Uint8Array(n);
    let pos = 0;
    for (const c of chunks) {
      out.set(c.subarray(0, Math.min(c.length, n - pos)), pos);
      pos += c.length;
      if (pos >= n) break;
    }
    return out;
  }

  /* Convert 32 LSB bits (Uint8Array) → float ∈ [0,1).  (unsigned!) */
  _bitsToFloat32(bits32) {
    let val = 0;
    for (let i = 0; i < 32; i++) {
      // Use arithmetic not bitwise to avoid signed 32‑bit overflow.
      val = val * 2 + bits32[i];
    }
    return val / 4294967296; // 2^32
  }

  /* ------------- public API ------------- */
  /* random(k)  → Float32Array of k uniform floats in [0,1) */
  async random(k = 1) {
    const needBits = k * 32;
    const bits = await this._collectBits(needBits);
    const out = new Float32Array(k);
    for (let i = 0; i < k; i++) {
      out[i] = this._bitsToFloat32(bits.subarray(i * 32, (i + 1) * 32));
    }
    return k === 1 ? out[0] : out;
  }

  /* randint(low, high, k=1) – high **exclusive** (Python‑style). */
  async randint(low, high = null, k = 1) {
    if (high === null) { high = low; low = 0; }
    if (high <= low) throw new RangeError("high must be > low");
    const range = high - low;
    const floats = await this.random(k);
    const arrayified = k === 1 ? [floats] : Array.from(floats);
    const ints = arrayified.map(f => low + Math.floor(f * range));
    return k === 1 ? ints[0] : ints;
  }

  /* randrange(start, stop, step=1) – single int. */
  async randrange(start, stop = null, step = 1) {
    if (step === 0) throw new RangeError("step must not be zero");
    if (stop === null) { stop = start; start = 0; }
    const n = Math.floor((stop - start + step - 1) / step);
    if (n <= 0) throw new RangeError("empty range");
    const idx = await this.randint(0, n);
    return start + idx * step;
  }

  /* choice(array) */
  async choice(arr) {
    if (!arr.length) throw new RangeError("Cannot choose from an empty array");
    const idx = await this.randint(0, arr.length);
    return arr[idx];
  }

  /* choices(array, k) – with replacement */
  async choices(arr, k = 1) {
    if (!arr.length) throw new RangeError("Cannot choose from an empty array");
    const idxs = await this.randint(0, arr.length, k);
    return idxs.map(i => arr[i]);
  }

  /* shuffle(array) – in‑place Fisher‑Yates */
  async shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = await this.randint(0, i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  /* sample(array, k) – without replacement */
  async sample(arr, k) {
    if (k < 0 || k > arr.length) throw new RangeError("Sample larger than population or negative");
    const pool = arr.slice();
    await this.shuffle(pool);
    return pool.slice(0, k);
  }
}
