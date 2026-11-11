// Idle hero: a hand skeleton that morphs between real gesture poses (the most typical HaGRIDv2 example of each),
// naming each one as it lands. It is the first thing a visitor sees before the camera is on, so it teaches the
// vocabulary instead of sitting there as an empty placeholder.
import { gestureInfo } from "./gestures-meta.js";
import { reducedMotion } from "./motion.js";

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const SEQUENCE = ["palm", "peace", "ok", "rock", "like", "call", "three_gun", "one", "thumb_index", "fist"];
const TIPS = [4, 8, 12, 16, 20];
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

function normalise(points) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const s = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1e-6);
  return points.map(([x, y]) => [(x - cx) / s, (y - cy) / s]);
}

export class PoseHero {
  /** canvas: the drawing surface; caption: { name, hint, index } elements (optional). */
  constructor(canvas, poses, caption = {}) {
    this.canvas = canvas;
    this.g = canvas.getContext("2d");
    this.caption = caption;
    this.ids = SEQUENCE.filter((id) => poses[id]?.hands?.length === 1);
    this.shapes = this.ids.map((id) => normalise(poses[id].hands[0]));
    this.i = 0;
    this.t0 = performance.now();
    this.running = false;
    this.trail = [];
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    this.label(0);
  }

  resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.w = this.canvas.clientWidth;
    this.h = this.canvas.clientHeight;
    this.canvas.width = Math.max(1, Math.round(this.w * dpr));
    this.canvas.height = Math.max(1, Math.round(this.h * dpr));
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  label(i) {
    const id = this.ids[i];
    const info = gestureInfo(id);
    const { name, hint, index, root } = this.caption;
    if (root) {
      root.classList.remove("swap");
      void root.offsetWidth;
      root.classList.add("swap");
    }
    if (name) name.textContent = info.name;
    if (hint) hint.textContent = info.hint;
    if (index) index.textContent = `${String(i + 1).padStart(2, "0")} / ${String(this.ids.length).padStart(2, "0")}`;
  }

  start() {
    if (this.running || !this.ids.length) return;
    this.running = true;
    const frame = (now) => {
      if (!this.running) return;
      this.draw(now);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  stop() {
    this.running = false;
  }

  draw(now) {
    const HOLD = 1900;
    const MORPH = reducedMotion.matches ? 1 : 750;
    const elapsed = now - this.t0;
    if (elapsed > HOLD + MORPH) {
      this.i = (this.i + 1) % this.ids.length;
      this.t0 = now;
      this.label(this.i);
    }
    const k = ease(Math.min(1, Math.max(0, (now - this.t0 - HOLD) / MORPH)));
    const a = this.shapes[this.i];
    const b = this.shapes[(this.i + 1) % this.shapes.length];
    const { g, w, h } = this;
    // Wide: the hand owns the right half of the stage. Narrow: it sits above the copy.
    const wide = w > 900 && w / h > 1.1;
    const size = wide ? Math.min(w * 0.4, h * 0.64) : Math.min(w * 0.62, h * 0.27);
    const ox = wide ? w * 0.66 : w / 2;
    const oy = wide ? h * 0.48 : Math.max(h * 0.22, 64 + size * 0.55);
    // A slow breathing drift so the hand reads as tracked, not drawn.
    const bob = reducedMotion.matches ? 0 : Math.sin(now / 1400) * size * 0.012;
    const sway = reducedMotion.matches ? 0 : Math.sin(now / 2300) * size * 0.01;
    const pts = a.map((p, j) => [
      ox + sway + (p[0] + (b[j][0] - p[0]) * k) * size,
      oy + bob + (p[1] + (b[j][1] - p[1]) * k) * size,
    ]);

    g.clearRect(0, 0, w, h);
    const color = this.i % 2 ? "#7fa8ff" : "#ffb25c";

    // Fingertip trail of the index finger: the signature of the motion-capture look.
    this.trail.push(pts[8]);
    if (this.trail.length > 28) this.trail.shift();
    g.lineCap = g.lineJoin = "round";
    for (let j = 1; j < this.trail.length; j++) {
      g.strokeStyle = color;
      g.globalAlpha = (j / this.trail.length) ** 2 * 0.5;
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(...this.trail[j - 1]);
      g.lineTo(...this.trail[j]);
      g.stroke();
    }
    g.globalAlpha = 1;

    // Bones: a soft glow pass, then the crisp stroke.
    g.strokeStyle = color;
    for (const [width, alpha] of [[size * 0.06, 0.035], [size * 0.024, 0.08], [size * 0.008, 1]]) {
      g.globalAlpha = alpha;
      g.lineWidth = Math.max(1.5, width);
      g.beginPath();
      for (const [p, q] of CONNECTIONS) {
        g.moveTo(...pts[p]);
        g.lineTo(...pts[q]);
      }
      g.stroke();
    }
    g.globalAlpha = 1;
    g.fillStyle = "#f5f5f7";
    pts.forEach(([x, y], j) => {
      g.beginPath();
      g.arc(x, y, (TIPS.includes(j) ? 0.014 : j === 0 ? 0.016 : 0.009) * size, 0, Math.PI * 2);
      g.fill();
    });
  }
}
