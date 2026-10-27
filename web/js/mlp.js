// Dependency-free MLP inference for models exported by scripts/export_web.py (BatchNorm already folded in).

function decode(b64) {
  const bytes = typeof atob === "function" ? Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)) : Buffer.from(b64, "base64");
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

// erf via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), for exact (non-tanh) GELU as in PyTorch.
function erf(x) {
  const s = Math.sign(x);
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return s * y;
}

const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

export class GestureModel {
  constructor(spec) {
    this.kind = spec.kind;
    this.classes = spec.classes;
    this.rejectIndex = spec.reject_index;
    this.temperature = spec.temperature;
    this.rejectBias = spec.reject_bias;
    this.inDim = spec.in_dim;
    this.mean = decode(spec.mean);
    this.std = decode(spec.std);
    this.layers = spec.layers.map((l) => ({ out: l.out, in: l.in, w: decode(l.w), b: decode(l.b), act: l.act }));
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
    return new GestureModel(await response.json());
  }

  logits(features) {
    return this.forward(features, this.layers.length);
  }

  /** L2-normalised activations of the last hidden layer: a learned hand-shape embedding (used for custom gestures). */
  embed(features) {
    const h = this.forward(features, this.layers.length - 1);
    const norm = Math.hypot(...h) || 1;
    return h.map((v) => v / norm);
  }

  forward(features, numLayers) {
    let x = new Float32Array(this.inDim);
    for (let i = 0; i < this.inDim; i++) x[i] = (features[i] - this.mean[i]) / this.std[i];
    for (const layer of this.layers.slice(0, numLayers)) {
      const y = new Float32Array(layer.out);
      for (let o = 0; o < layer.out; o++) {
        let sum = layer.b[o];
        const row = o * layer.in;
        for (let i = 0; i < layer.in; i++) sum += layer.w[row + i] * x[i];
        y[o] = layer.act === "gelu" ? gelu(sum) : sum;
      }
      x = y;
    }
    return x;
  }

  /** Calibrated class probabilities (temperature + rejection bias), as GestureClassifier.predict_proba. */
  predictProba(features, rejectBiasOffset = 0) {
    const z = Array.from(this.logits(features), (v) => v / this.temperature);
    z[this.rejectIndex] += this.rejectBias + rejectBiasOffset;
    const max = Math.max(...z);
    const exp = z.map((v) => Math.exp(v - max));
    const total = exp.reduce((a, b) => a + b, 0);
    return exp.map((v) => v / total);
  }
}
