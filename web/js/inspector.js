// Hand inspector: live measurements from MediaPipe's world landmarks (metres, centred on the hand), shown beside the
// 3D lab. Finger bend per finger, hand span, pinch gap and palm length. Values ease every frame (springs).
import { Spring } from "./motion.js";

const FINGERS = [
  { name: "Thumb", chain: [0, 1, 2, 3, 4], max: 120 },
  { name: "Index", chain: [0, 5, 6, 7, 8], max: 250 },
  { name: "Middle", chain: [0, 9, 10, 11, 12], max: 250 },
  { name: "Ring", chain: [0, 13, 14, 15, 16], max: 250 },
  { name: "Little", chain: [0, 17, 18, 19, 20], max: 250 },
];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (v) => Math.hypot(v[0], v[1], v[2]);
const dist = (a, b) => len(sub(a, b));
function angle(u, v) {
  const d = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / Math.max(len(u) * len(v), 1e-9);
  return (Math.acos(Math.min(1, Math.max(-1, d))) * 180) / Math.PI;
}

/** Measurements of one hand from 21 world landmarks [x, y, z] in metres. */
export function measure(world) {
  const bend = FINGERS.map(({ chain }) => {
    let total = 0;
    for (let i = 1; i < chain.length - 1; i++) total += angle(sub(world[chain[i]], world[chain[i - 1]]), sub(world[chain[i + 1]], world[chain[i]]));
    return total;
  });
  return {
    bend, // degrees of total bend along each finger: 0 = straight
    span: dist(world[4], world[20]) * 100, // cm, thumb tip to little fingertip
    pinch: dist(world[4], world[8]) * 1000, // mm, thumb tip to index tip
    palm: dist(world[0], world[9]) * 100, // cm, wrist to middle knuckle
  };
}

export class HandInspector {
  constructor(root) {
    this.root = root;
    root.innerHTML = `
      <div class="insp-head"><b id="insp-title">No hand</b><span id="insp-chan"></span></div>
      <ol class="insp-fingers">${FINGERS.map((f, i) => `<li><span>${f.name}</span><i><b data-f="${i}"></b></i><output data-o="${i}">–</output></li>`).join("")}</ol>
      <dl class="insp-stats">
        <div><dt>Span</dt><dd><b data-s="span">–</b> cm</dd></div>
        <div><dt>Pinch gap</dt><dd><b data-s="pinch">–</b> mm</dd></div>
        <div><dt>Palm</dt><dd><b data-s="palm">–</b> cm</dd></div>
      </dl>
      <p class="insp-note">Measured in 3D from the landmark model's real-world coordinates, not pixels, so the numbers hold at any distance from the camera.</p>`;
    this.bars = [...root.querySelectorAll("[data-f]")];
    this.outs = [...root.querySelectorAll("[data-o]")];
    this.stats = Object.fromEntries([...root.querySelectorAll("[data-s]")].map((el) => [el.dataset.s, el]));
    this.bend = FINGERS.map(() => new Spring(0, { k: 140, zeta: 1 }));
    this.values = { span: new Spring(0, { k: 90, zeta: 1 }), pinch: new Spring(0, { k: 120, zeta: 1 }), palm: new Spring(0, { k: 90, zeta: 1 }) };
    this.present = false;
  }

  /** hand: engine hand with .world and .handedness, or null. */
  update(hand, name, color) {
    this.present = Boolean(hand?.world);
    this.root.classList.toggle("idle", !this.present);
    this.root.style.setProperty("--c", color);
    this.root.querySelector("#insp-title").textContent = this.present ? name : "No hand";
    this.root.querySelector("#insp-chan").textContent = this.present ? `${hand.handedness} hand` : "Raise a hand to measure it";
    if (!this.present) return;
    const m = measure(hand.world);
    m.bend.forEach((v, i) => this.bend[i].set(Math.min(1, v / FINGERS[i].max)));
    m.bend.forEach((v, i) => (this.outs[i].textContent = `${Math.round(v)}°`));
    for (const k of ["span", "pinch", "palm"]) this.values[k].set(m[k]);
  }

  step(dt) {
    this.bend.forEach((s, i) => (this.bars[i].style.transform = `scaleX(${Math.max(0, Math.min(1, s.step(dt))).toFixed(4)})`));
    for (const [k, s] of Object.entries(this.values)) {
      const v = s.step(dt);
      const text = !this.present ? "–" : k === "pinch" ? Math.round(v).toString() : v.toFixed(1);
      if (this.stats[k].textContent !== text) this.stats[k].textContent = text;
    }
  }
}
