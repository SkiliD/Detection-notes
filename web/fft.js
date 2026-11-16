// Small radix-2 FFT helpers: the browser has no numpy, and both the pitch
// reader and the attack tracker need one transform per frame/block.
// Kept dependency-free so the page stays a static folder GitHub Pages can
// serve as-is (no build step, no CDN).

/** In-place iterative radix-2 FFT. `re`/`im` must share a power-of-two length. */
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k];
        const ai = im[i + k];
        const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ar + br;
        im[i + k] = ai + bi;
        re[i + k + len / 2] = ar - br;
        im[i + k + len / 2] = ai - bi;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

function nextPow2(n) {
  let size = 1;
  while (size < n) size <<= 1;
  return size;
}

/**
 * Power spectrum of `x` (zero-padded to `size`), the first `size/2 + 1` bins.
 * Same shape numpy's `rfft(x, size)` produces, so `attacks.js` can mirror
 * `AttackTracker.flatness` bin for bin.
 */
export function rfftPower(x, size) {
  const n = size ?? nextPow2(x.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re.set(x.subarray(0, Math.min(x.length, n)));
  fft(re, im);
  const power = new Float64Array(n / 2 + 1);
  for (let k = 0; k < power.length; k++) {
    power[k] = re[k] * re[k] + im[k] * im[k];
  }
  return power;
}

/**
 * Linear autocorrelation r[0..width-1] via the FFT, normalized like
 * numpy's `irfft(rfft(x) * conj(rfft(x)))` (which carries the 1/N).
 * `difference` in `yin.js` subtracts `2 * r[tau]`, so the scaling must match.
 */
export function autocorrelation(x) {
  const width = x.length;
  let size = 1;
  while (size < 2 * width) size <<= 1;
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  re.set(x);
  fft(re, im);
  for (let k = 0; k < size; k++) {
    const p = re[k] * re[k] + im[k] * im[k];
    re[k] = p;
    im[k] = 0;
  }
  fft(re, im); // inverse: conj(conj) — the sign flip of the twiddles
  // running the forward transform twice yields N * conj(ifft(z)); z is real
  // and symmetric here, so the real part scaled by 1/N is the correlation
  const out = new Float64Array(width);
  for (let tau = 0; tau < width; tau++) out[tau] = re[tau] / size;
  // re is already the inverse-transform real part times `size`, because the
  // second forward pass conjugates the phase: divide once, above.
  return out;
}
