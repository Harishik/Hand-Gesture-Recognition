// Idle hero: a hand skeleton that morphs between real gesture poses (the most typical HaGRIDv2 example of each),
// naming each one as it lands. It is the first thing a visitor sees before the camera is on, so it teaches the
// vocabulary instead of sitting there as an empty placeholder.
//
// Layout is measured, not guessed: the hand is fitted into the largest free rectangle that does not touch the copy
// or the caption (the elements passed as `avoid`), so it can never overlap text at any window size. If there is not
// enough free space, the hand is simply not drawn.
import { gestureInfo } from "./gestures-meta.js";
import { reducedMotion } from "./motion.js";

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const SEQUENCE = ["palm", "peace", "ok", "rock", "like", "call", "three_gun", "one", "thumb_index", "fist"];
const TIPS = [4, 8, 12, 16, 20];
const COLORS = [[255, 178, 92], [127, 168, 255]]; // right-hand amber, left-hand cobalt
const HOLD = 2.2; // s
const MORPH = 0.9; // s
const BAR = 64; // px reserved for the floating top bar
const GAP = 40; // px kept clear around text
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
  /** canvas: drawing surface. caption: { name, hint, root } elements. avoid: elements the hand must never cover. */
  constructor(canvas, poses, caption = {}, avoid = []) {
    this.canvas = canvas;
    this.g = canvas.getContext("2d");
    this.caption = caption;
    this.avoid = avoid.filter(Boolean);
    this.ids = SEQUENCE.filter((id) => poses[id]?.hands?.length === 1);
    this.shapes = this.ids.map((id) => normalise(poses[id].hands[0]));
    this.i = 0;
    this.clock = 0; // own clock, advanced by clamped frame time: a stall pauses the hand, it never jumps
    this.labelled = 0;
    this.alpha = 0;
    this.running = false;
    this.trail = [];
    const ro = new ResizeObserver(() => this.layout());
    ro.observe(canvas);
    this.avoid.forEach((el) => ro.observe(el));
    this.layout();
    this.label(0);
  }

  layout() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.w = this.canvas.clientWidth;
    this.h = this.canvas.clientHeight;
    this.canvas.width = Math.max(1, Math.round(this.w * dpr));
    this.canvas.height = Math.max(1, Math.round(this.h * dpr));
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.box = this.freeBox();
  }

  /** The largest rectangle on the canvas clear of every `avoid` element (plus a gap), as { x, y, w, h }. */
  freeBox() {
    const origin = this.canvas.getBoundingClientRect();
    const rects = this.avoid
      .filter((el) => el.offsetParent !== null)
      .map((el) => {
        const r = el.getBoundingClientRect();
        return { l: r.left - origin.left - GAP, t: r.top - origin.top - GAP, r: r.right - origin.left + GAP, b: r.bottom - origin.top + GAP };
      });
    const top = Math.max(BAR, 0);
    const pad = 24;
    const candidates = [];
    // Everything to the right of the text column (elements whose centre is in the left half).
    const right = Math.max(pad, ...rects.filter((r) => (r.l + r.r) / 2 < this.w / 2).map((r) => r.r));
    candidates.push({ x: right, y: top, w: this.w - pad - right, h: this.h - top - pad });
    const above = Math.min(this.h - pad, ...rects.map((r) => r.t)); // everything above the text
    candidates.push({ x: pad, y: top, w: this.w - 2 * pad, h: above - top });
    // The right column must also stay clear of any element that reaches into it (e.g. the pose caption).
    const fit = candidates.map((c) => {
      for (const r of rects) {
        const overlapsX = r.r > c.x && r.l < c.x + c.w;
        const overlapsY = r.b > c.y && r.t < c.y + c.h;
        if (overlapsX && overlapsY) c.h = Math.max(0, Math.min(c.h, r.t - c.y));
      }
      return c;
    });
    return fit.reduce((best, c) => (Math.min(c.w, c.h) > Math.min(best.w, best.h) ? c : best));
  }

  label(i) {
    const info = gestureInfo(this.ids[i]);
    const { name, hint, root } = this.caption;
    if (root) {
      root.classList.remove("swap");
      void root.offsetWidth;
      root.classList.add("swap");
    }
    if (name) name.textContent = info.name;
    if (hint) hint.textContent = info.hint;
  }

  start() {
    if (this.running || !this.ids.length) return;
    this.running = true;
    let last = performance.now();
    const frame = (now) => {
      if (!this.running) return;
      const dt = Math.min((now - last) / 1000, 1 / 30);
      last = now;
      this.draw(dt, now);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  stop() {
    this.running = false;
  }

  draw(dt, now) {
    this.clock += dt;
    const morph = reducedMotion.matches ? 0.001 : MORPH;
    if (this.clock > HOLD + morph) {
      this.clock -= HOLD + morph;
      this.i = (this.i + 1) % this.ids.length;
    }
    const k = ease(Math.min(1, Math.max(0, (this.clock - HOLD) / morph)));
    const next = (this.i + 1) % this.ids.length;
    // The caption changes half-way through the morph, when the shape already reads as the next pose.
    const shown = k > 0.5 ? next : this.i;
    if (shown !== this.labelled) {
      this.labelled = shown;
      this.label(shown);
    }
    const a = this.shapes[this.i];
    const b = this.shapes[next];
    const { g, w, h, box } = this;
    g.clearRect(0, 0, w, h);

    // The pose is normalised to a unit box; glow and joints add ~10% around it.
    const size = Math.min(box.w, box.h) * 0.8;
    if (size < 120) return; // not enough clear space: show no hand rather than one that crowds the text
    this.alpha = Math.min(1, this.alpha + dt * 2.5);
    const ox = box.x + box.w / 2;
    const oy = box.y + box.h / 2;
    const still = reducedMotion.matches;
    const bob = still ? 0 : Math.sin(now / 1400) * size * 0.012;
    const sway = still ? 0 : Math.sin(now / 2300) * size * 0.01;
    const pts = a.map((p, j) => [ox + sway + (p[0] + (b[j][0] - p[0]) * k) * size, oy + bob + (p[1] + (b[j][1] - p[1]) * k) * size]);

    // Colour crossfades with the shape instead of switching on a frame.
    const ca = COLORS[this.i % 2];
    const cb = COLORS[next % 2];
    const color = `rgb(${ca.map((v, c) => Math.round(v + (cb[c] - v) * k)).join(",")})`;

    g.globalAlpha = this.alpha;
    this.trail.push(pts[8]);
    if (this.trail.length > 28) this.trail.shift();
    g.lineCap = g.lineJoin = "round";
    g.strokeStyle = color;
    g.lineWidth = 1.5;
    for (let j = 1; j < this.trail.length; j++) {
      g.globalAlpha = this.alpha * (j / this.trail.length) ** 2 * 0.5;
      g.beginPath();
      g.moveTo(...this.trail[j - 1]);
      g.lineTo(...this.trail[j]);
      g.stroke();
    }
    for (const [width, alpha] of [[size * 0.06, 0.035], [size * 0.024, 0.08], [size * 0.008, 1]]) {
      g.globalAlpha = this.alpha * alpha;
      g.lineWidth = Math.max(1.5, width);
      g.beginPath();
      for (const [p, q] of CONNECTIONS) {
        g.moveTo(...pts[p]);
        g.lineTo(...pts[q]);
      }
      g.stroke();
    }
    g.globalAlpha = this.alpha;
    g.fillStyle = "#f5f5f7";
    pts.forEach(([x, y], j) => {
      g.beginPath();
      g.arc(x, y, (TIPS.includes(j) ? 0.014 : j === 0 ? 0.016 : 0.009) * size, 0, Math.PI * 2);
      g.fill();
    });
    g.globalAlpha = 1;
  }
}
