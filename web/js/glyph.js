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
