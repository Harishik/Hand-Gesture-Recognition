// Turns per-frame hand landmarks into stable gestures and events:
// per-hand classification with temporal smoothing, two-hand gestures, dwell-to-confirm events, swipes from palm
// motion, and a pinch-driven air cursor.

import { featurize, pairFeatures } from "./features.js";
import { OneEuroFilter, emaVector } from "./filters.js";

const OPEN_HANDS = new Set(["palm", "stop", "stop_inverted", "four"]);
const PALM_POINTS = [0, 5, 9, 13, 17];

export const DEFAULT_SETTINGS = {
  smoothing: 0.6, // weight on the previous probabilities (0 = none)
  minConfidence: 0.6, // below this the hand shows as no_gesture
  holdMs: 450, // a gesture must be held this long to fire an event
  rejectBias: 0, // extra bias on no_gesture: + = stricter, - = more eager
  twoHand: true,
  swipes: true,
  cursor: true,
  mirror: true,
};

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export class GestureEngine {
  constructor(handModel, pairModel, customGestures) {
    this.hand = handModel;
    this.pair = pairModel;
    this.custom = customGestures;
    this.settings = { ...DEFAULT_SETTINGS };
    this.tracks = new Map(); // key -> { probs, dwell, history, pinching }
    this.pairTrack = { probs: null, dwell: null };
    this.cursorX = new OneEuroFilter({ minCutoff: 1.0, beta: 0.04 });
    this.cursorY = new OneEuroFilter({ minCutoff: 1.0, beta: 0.04 });
    this.lastSwipe = -Infinity;
  }

  /**
   * hands: [{ landmarks: [[x, y], ...21], handedness: "Left" | "Right" }] in raw image coordinates.
   * Returns { hands, pair, events, cursor, classifyMs }.
   */
  /** Matches hands to existing tracks by nearest palm position (greedy); unmatched hands get a new track. */
  assignTracks(hands) {
    const free = [...this.tracks.entries()].filter(([, tr]) => tr.anchor);
    return hands.map((h) => {
      const p = h.landmarks[9];
      let best = -1;
      let bestD = 0.2; // max palm travel between frames, in normalised image units
      free.forEach(([, tr], j) => {
        const d = Math.hypot(tr.anchor[0] - p[0], tr.anchor[1] - p[1]);
        if (d < bestD) { bestD = d; best = j; }
      });
      if (best >= 0) return free.splice(best, 1)[0][0];
      this.nextTrack = (this.nextTrack ?? 0) + 1;
      return `hand${this.nextTrack}`;
    });
  }

  /**
   * Label hysteresis: the shown label only changes when the challenger clearly leads (by a margin) or has been on
   * top for a short time. Stops one-frame flickers while a hand moves between poses.
   */
  stableLabel(track, top, s, tMs) {
    const want = top[0].p >= s.minConfidence ? top[0].id : "no_gesture";
    const cur = track.shown ?? want;
    if (want === cur) { track.shown = cur; track.challenger = null; return cur; }
    const idx = this.hand.classes.indexOf(cur);
    const curP = idx >= 0 ? track.probs[idx] : 0;
    const margin = top[0].p - curP;
    if (!track.challenger || track.challenger.id !== want) track.challenger = { id: want, since: tMs };
    if (margin > 0.35 || tMs - track.challenger.since >= 150) {
      track.shown = want;
      track.challenger = null;
    }
    return track.shown;
  }

  update(hands, tMs, aspect) {
    const t0 = performance.now();
    const s = this.settings;
    const events = [];
    const seen = new Set();

    // Follow each hand by where it is, not by MediaPipe's Left/Right label: that label often flips mid-transition,
    // which used to reset the smoothing and let a single raw frame through (e.g. a fist briefly read as another class).
    const keys = this.assignTracks(hands);
    const out = hands.map((h, i) => {
      const key = keys[i];
      seen.add(key);
      const track = this.tracks.get(key) ?? { probs: null, dwell: null, history: [], pinching: false };
      this.tracks.set(key, track);
      track.anchor = h.landmarks[9];

      const features = featurize(h.landmarks, aspect);
      const probs = this.hand.predictProba(features, s.rejectBias);
      track.probs = emaVector(track.probs, probs, s.smoothing);
      const order = track.probs.map((p, c) => [p, c]).sort((a, b) => b[0] - a[0]);
      const top = order.slice(0, 5).map(([p, c]) => ({ id: this.hand.classes[c], p }));

      let label = this.stableLabel(track, top, s, tMs);
      let confidence = label === "no_gesture" ? 1 - top[0].p : track.probs[this.hand.classes.indexOf(label)] ?? top[0].p;
      if (label === top[0].id) confidence = top[0].p;
      const custom = this.custom.items.length ? this.custom.match(this.hand.embed(features)) : null;
      if (custom) {
        label = custom.id;
        confidence = custom.similarity;
      }

      const display = h.landmarks.map(([x, y]) => [s.mirror ? 1 - x : x, y]);
      const palm = Math.max(dist(display[0], display[9]) * 1, 1e-6);
      const result = { key, handedness: h.handedness, landmarks: display, features, top, label, confidence, custom };

      this.trackMotion(track, result, display, palm, tMs, aspect, events);
      this.trackPinch(track, result, display, palm, aspect, events);
      return result;
    });

    for (const key of [...this.tracks.keys()]) if (!seen.has(key)) this.tracks.delete(key);

    // Two-hand gestures override the individual hands while confidently recognised.
    let pair = null;
    if (s.twoHand && this.pair && out.length >= 2) {
      const [a, b] = out;
      const probs = this.pair.predictProba(pairFeatures(hands[0].landmarks, hands[1].landmarks, aspect));
      this.pairTrack.probs = emaVector(this.pairTrack.probs, probs, s.smoothing);
      const best = this.pairTrack.probs.reduce((bi, p, i, arr) => (p > arr[bi] ? i : bi), 0);
      const id = this.pair.classes[best];
      if (id !== "no_gesture" && this.pairTrack.probs[best] >= s.minConfidence) {
        pair = { id, confidence: this.pairTrack.probs[best], hands: [a.key, b.key] };
      }
    } else {
      this.pairTrack.probs = null;
    }

    // Dwell-to-confirm: a gesture becomes an event after being held for holdMs (once per hold).
    if (pair) {
      this.dwell(this.pairTrack, pair.id, tMs, events, { source: "two-hand" });
      pair.hold = this.pairTrack.dwell?.progress ?? 0;
      for (const h of out) h.pair = pair;
    } else {
      this.pairTrack.dwell = null;
    }
    for (const h of out) {
      if (pair) continue;
      const track = this.tracks.get(h.key);
      this.dwell(track, h.label, tMs, events, { source: h.handedness });
      h.hold = track.dwell?.progress ?? 0;
    }

    return {
      hands: out,
      pair,
      events,
      cursor: this.cursor(out, tMs),
      classifyMs: performance.now() - t0,
    };
  }

  dwell(track, id, tMs, events, meta) {
    if (id === "no_gesture") {
      track.dwell = null;
      return;
    }
    if (!track.dwell || track.dwell.id !== id) track.dwell = { id, since: tMs, fired: false, progress: 0 };
    const d = track.dwell;
    d.progress = Math.min(1, (tMs - d.since) / this.settings.holdMs);
    if (d.progress >= 1 && !d.fired) {
      d.fired = true;
      events.push({ type: "gesture", id, t: tMs, ...meta });
    }
  }

  trackMotion(track, result, display, palm, tMs, aspect, events) {
    const cx = PALM_POINTS.reduce((a, i) => a + display[i][0], 0) / PALM_POINTS.length;
    const cy = PALM_POINTS.reduce((a, i) => a + display[i][1], 0) / PALM_POINTS.length;
    track.history.push({ t: tMs, x: cx * aspect, y: cy, palm: palm * Math.max(aspect, 1) });
    while (track.history.length && tMs - track.history[0].t > 320) track.history.shift();
    if (!this.settings.swipes || track.history.length < 4 || tMs - this.lastSwipe < 700) return;
    if (!OPEN_HANDS.has(result.top[0].id)) return;

    const first = track.history[0];
    const last = track.history.at(-1);
    const scale = Math.max((first.palm + last.palm) / 2, 1e-6);
    const dx = (last.x - first.x) / scale;
    const dy = (last.y - first.y) / scale;
    let id = null;
    if (Math.abs(dx) > 2.2 && Math.abs(dx) > 1.8 * Math.abs(dy)) id = dx > 0 ? "swipe_right" : "swipe_left";
    else if (Math.abs(dy) > 2.0 && Math.abs(dy) > 1.8 * Math.abs(dx)) id = dy > 0 ? "swipe_down" : "swipe_up";
    if (id) {
      this.lastSwipe = tMs;
      track.history = [];
      events.push({ type: "gesture", id, t: tMs, source: result.handedness });
    }
  }

  trackPinch(track, result, display, palm, aspect, events) {
    // Thumb tip to index tip, relative to palm length (isotropic units), with hysteresis.
    const iso = (p) => [p[0] * aspect, p[1]];
    const ratio = dist(iso(display[4]), iso(display[8])) / Math.max(dist(iso(display[0]), iso(display[9])), 1e-6);
    const was = track.pinching;
    track.pinching = was ? ratio < 0.45 : ratio < 0.28;
    result.pinch = { active: track.pinching, ratio };
    if (track.pinching !== was) events.push({ type: track.pinching ? "pinch_start" : "pinch_end", source: result.handedness });
  }

  cursor(hands, tMs) {
    if (!this.settings.cursor || !hands.length) {
      this.cursorX.reset();
      this.cursorY.reset();
      return null;
    }
    // The nearest hand (largest on screen) drives the cursor: index fingertip, or the pinch point while pinching.
    const size = (h) => dist(h.landmarks[0], h.landmarks[9]);
    const h = hands.reduce((a, b) => (size(b) > size(a) ? b : a));
    const p = h.pinch?.active
      ? [(h.landmarks[4][0] + h.landmarks[8][0]) / 2, (h.landmarks[4][1] + h.landmarks[8][1]) / 2]
      : h.landmarks[8];
    // The central 70% of the camera view maps to the whole screen, so corners are reachable.
    const map = (v) => Math.min(1, Math.max(0, (v - 0.15) / 0.7));
    return {
      x: this.cursorX.filter(map(p[0]), tMs),
      y: this.cursorY.filter(map(p[1]), tMs),
      pinching: Boolean(h.pinch?.active),
      hand: h.key,
    };
  }
}
