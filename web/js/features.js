// Hand-landmark features: a line-for-line port of src/hgr/features.py and src/hgr/pairs.py.
// Landmarks are arrays of 21 [x, y] pairs in image-normalised coordinates. Kept DOM-free so Node can test parity.

export const NUM_LANDMARKS = 21;
const WRIST = 0;
const MIDDLE_MCP = 9;
const FINGERTIPS = [4, 8, 12, 16, 20];
const FINGER_CHAINS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20],
];
const TIP_PAIRS = [];
FINGERTIPS.forEach((a, i) => FINGERTIPS.slice(i + 1).forEach((b) => TIP_PAIRS.push([a, b])));
const JOINT_TRIPLES = FINGER_CHAINS.flatMap((chain) => [0, 1, 2].map((i) => chain.slice(i, i + 3)));

export const FEATURE_DIM = NUM_LANDMARKS * 2 + JOINT_TRIPLES.length + TIP_PAIRS.length; // 67
export const PAIR_FEATURE_DIM = 2 * NUM_LANDMARKS * 2 + 2 * FEATURE_DIM + 3; // 221

/** x rescaled by aspect (width / height), wrist at the origin, divided by palm length. */
export function normalizeLandmarks(landmarks, aspect = 1) {
  const pts = landmarks.map(([x, y]) => [Math.fround(x * aspect), Math.fround(y)]);
  const [wx, wy] = pts[WRIST];
  for (const p of pts) {
    p[0] -= wx;
    p[1] -= wy;
  }
  const palm = Math.max(Math.hypot(pts[MIDDLE_MCP][0], pts[MIDDLE_MCP][1]), 1e-6);
  return pts.map(([x, y]) => [x / palm, y / palm]);
}

function jointAngles(pts) {
  return JOINT_TRIPLES.map(([a, b, c]) => {
    const v1 = [pts[b][0] - pts[a][0], pts[b][1] - pts[a][1]];
    const v2 = [pts[c][0] - pts[b][0], pts[c][1] - pts[b][1]];
    const denom = Math.max(Math.hypot(...v1) * Math.hypot(...v2), 1e-6);
    const cos = (v1[0] * v2[0] + v1[1] * v2[1]) / denom;
    return Math.acos(Math.min(1, Math.max(-1, cos)));
  });
}

function fingertipDistances(pts) {
  return TIP_PAIRS.map(([a, b]) => Math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]));
}

/** Features from already-normalised points. */
export function featuresFromPoints(pts) {
  const out = new Float32Array(FEATURE_DIM);
  let k = 0;
  for (const [x, y] of pts) {
    out[k++] = x;
    out[k++] = y;
  }
  for (const v of jointAngles(pts)) out[k++] = v;
  for (const v of fingertipDistances(pts)) out[k++] = v;
  return out;
}

export function featurize(landmarks, aspect = 1) {
  return featuresFromPoints(normalizeLandmarks(landmarks, aspect));
}

/** Two hands (each 21 [x, y]) -> pair features. Hand order doesn't matter. */
export function pairFeatures(handA, handB, aspect = 1) {
  let a = handA.map(([x, y]) => [Math.fround(x * aspect), Math.fround(y)]);
  let b = handB.map(([x, y]) => [Math.fround(x * aspect), Math.fround(y)]);
  if (a[WRIST][0] > b[WRIST][0]) [a, b] = [b, a]; // left-most wrist first

  const cx = (a[WRIST][0] + b[WRIST][0]) / 2;
  const cy = (a[WRIST][1] + b[WRIST][1]) / 2;
  const palmA = Math.hypot(a[MIDDLE_MCP][0] - a[WRIST][0], a[MIDDLE_MCP][1] - a[WRIST][1]);
  const palmB = Math.hypot(b[MIDDLE_MCP][0] - b[WRIST][0], b[MIDDLE_MCP][1] - b[WRIST][1]);
  const scale = Math.max((palmA + palmB) / 2, 1e-6);

  const out = new Float32Array(PAIR_FEATURE_DIM);
  let k = 0;
  for (const hand of [a, b]) {
    for (const [x, y] of hand) {
      out[k++] = (x - cx) / scale;
      out[k++] = (y - cy) / scale;
    }
  }
  for (const hand of [a, b]) {
    out.set(featuresFromPoints(normalizeLandmarks(hand, 1)), k);
    k += FEATURE_DIM;
  }
  const dx = (b[WRIST][0] - a[WRIST][0]) / scale;
  const dy = (b[WRIST][1] - a[WRIST][1]) / scale;
  out[k++] = dx;
  out[k++] = dy;
  out[k++] = Math.hypot(dx, dy);
  return out;
}
