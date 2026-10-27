// Pose glyphs: each gesture drawn as the skeleton of its most typical HaGRIDv2 example (web/models/poses.json).

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

let poses = {};
export async function loadPoses(url = "models/poses.json") {
  poses = await (await fetch(url)).json();
  return poses;
}

/** Draws the gesture's pose skeleton centred in the box (x, y, size) of a 2D canvas context. */
export function drawGlyph(g, id, x, y, size, { color = "#ECE8E1", width = 2, joints = true } = {}) {
  const hands = poses[id]?.hands;
  if (!hands) {
    g.strokeStyle = color;
    g.lineWidth = width;
    g.beginPath();
    g.arc(x + size / 2, y + size / 2, size * 0.18, 0, Math.PI * 2);
    g.stroke();
    return;
  }
  const all = hands.flat();
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const s = (size * 0.8) / Math.max(x1 - x0, y1 - y0, 1e-6);
  const ox = x + (size - (x1 - x0) * s) / 2 - x0 * s;
  const oy = y + (size - (y1 - y0) * s) / 2 - y0 * s;
  g.save();
  g.strokeStyle = color;
  g.fillStyle = color;
  g.lineWidth = width;
  g.lineCap = "round";
  g.lineJoin = "round";
  for (const hand of hands) {
    const p = hand.map(([px, py]) => [ox + px * s, oy + py * s]);
    g.beginPath();
    for (const [a, b] of CONNECTIONS) {
      g.moveTo(...p[a]);
      g.lineTo(...p[b]);
    }
    g.stroke();
    if (joints) {
      for (const i of [4, 8, 12, 16, 20]) {
        g.beginPath();
        g.arc(p[i][0], p[i][1], width * 1.3, 0, Math.PI * 2);
        g.fill();
      }
    }
  }
  g.restore();
}

const ARROWS = { swipe_left: [-1, 0], swipe_right: [1, 0], swipe_up: [0, -1], swipe_down: [0, 1] };
const urlCache = new Map();

/** Data URL of a gesture glyph, cached. Motion gestures get an arrow, custom gestures a diamond, others a ring. */
export function glyphURL(id, size = 28, color = "#ece8e1") {
  const key = `${id}|${size}|${color}`;
  if (urlCache.has(key)) return urlCache.get(key);
  const dpr = 2;
  const c = document.createElement("canvas");
  c.width = c.height = size * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  const m = size / 2;
  g.strokeStyle = g.fillStyle = color;
  g.lineWidth = Math.max(1.4, size / 18);
  g.lineCap = g.lineJoin = "round";
  if (ARROWS[id]) {
    const [dx, dy] = ARROWS[id];
    const r = size * 0.3;
    g.beginPath();
    g.moveTo(m - dx * r, m - dy * r);
    g.lineTo(m + dx * r, m + dy * r);
    g.moveTo(m + dx * r - (dx + dy) * r * 0.45, m + dy * r - (dy - dx) * r * 0.45);
    g.lineTo(m + dx * r, m + dy * r);
    g.lineTo(m + dx * r - (dx - dy) * r * 0.45, m + dy * r - (dy + dx) * r * 0.45);
    g.stroke();
  } else if (id?.startsWith("custom:")) {
    g.beginPath();
    g.moveTo(m, size * 0.22);
    g.lineTo(size * 0.78, m);
    g.lineTo(m, size * 0.78);
    g.lineTo(size * 0.22, m);
    g.closePath();
    g.stroke();
  } else if (id === "pinch") {
    g.beginPath();
    g.arc(m, m, size * 0.12, 0, Math.PI * 2);
    g.fill();
    g.beginPath();
    g.arc(m, m, size * 0.28, 0, Math.PI * 2);
    g.stroke();
  } else {
    drawGlyph(g, id, 0, 0, size, { color, width: Math.max(1.2, size / 24), joints: size >= 40 });
  }
  const url = c.toDataURL();
  urlCache.set(key, url);
  return url;
}

/** A small canvas element showing a gesture glyph (crisp on high-DPI screens). */
export function glyphCanvas(id, cssSize, options) {
  const c = document.createElement("canvas");
  const dpr = Math.min(devicePixelRatio || 1, 2);
  c.width = c.height = Math.round(cssSize * dpr);
  c.style.width = c.style.height = `${cssSize}px`;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  drawGlyph(g, id, 0, 0, cssSize, options);
  return c;
}
