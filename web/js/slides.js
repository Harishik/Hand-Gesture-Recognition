// Live explainer slides: each slide is one stage of the pipeline, shown running on the visitor's own hand.
// Swipe left / right (or the arrow keys) to move between them. Only the visible slide is drawn each frame.
import { gestureInfo } from "./gestures-meta.js";
import { approach, Spring } from "./motion.js";

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const PALM = [0, 5, 9, 13, 17];
const CHANNEL = { Left: "#7fa8ff", Right: "#ffb25c" };
const SWIPE_AT = 2.2; // the engine's threshold: palm-widths travelled within 320 ms

export const SLIDE_COUNT = 5;

const COPY = [
  ["Step 1 · Landmarks", "21 points per hand", "MediaPipe finds each hand and places 21 landmarks on it, in a worker thread, every camera frame."],
  ["Step 2 · Classify", "67 features in, 33 gestures out", "Angles and distances between landmarks feed two small neural networks trained on 1.3M hands."],
  ["Step 3 · Confirm", "Hold to trigger", "A gesture only fires after it is held steadily, so passing shapes never set off an action."],
  ["Step 4 · Motion", "Swipes by speed", "Palm travel over 320 ms, measured in palm widths, so a flick works at any distance."],
  ["Step 5 · Private", "Nothing leaves the tab", "Video, landmarks and predictions stay in this browser. There is no server to send them to."],
];

export class LiveSlides {
  constructor(track, { labelInfo, icon, stats }) {
    this.track = track;
    this.labelInfo = labelInfo;
    this.stats = stats; // () => { frames, landmarkMs, mode, delegate }
    this.active = 0;
    this.state = null;
    this.lastEvent = null;
    this.history = [];
    track.innerHTML = COPY.map(
      ([kicker, title, body], i) => `<section class="slide" data-i="${i}" role="group" aria-roledescription="slide" aria-label="${i + 1} of ${COPY.length}: ${title}">
        <div class="slide-visual" data-v="${i}"></div>
        <div class="slide-copy"><span class="slide-kicker">${kicker}</span><h4>${title}</h4><p>${body}</p></div>
      </section>`,
    ).join("");
    const v = (i) => track.querySelector(`[data-v="${i}"]`);

    // 1 · landmarks: a live skeleton with numbered fingertips.
    this.lmCanvas = document.createElement("canvas");
    v(0).append(this.lmCanvas);
    this.lmPts = null;

    // 2 · classify: the model's top three, as bars.
    v(1).innerHTML = `<ol class="sv-probs">${[0, 1, 2].map(() => `<li><span class="sv-name"></span><span class="sv-pct"></span><i><b></b></i></li>`).join("")}</ol>`;
    this.probRows = [...v(1).querySelectorAll("li")].map((li) => ({ li, name: li.querySelector(".sv-name"), pct: li.querySelector(".sv-pct"), bar: li.querySelector("b"), s: new Spring(0, { k: 160, zeta: 1 }) }));
    this.icon = icon;

    // 3 · confirm: a hold ring around the current gesture.
    v(2).innerHTML = `<div class="sv-hold"><svg viewBox="0 0 120 120" aria-hidden="true"><circle class="t" cx="60" cy="60" r="52"/><circle class="a" cx="60" cy="60" r="52" pathLength="100"/></svg><div><b class="sv-hold-name">–</b><span class="sv-hold-last">Nothing confirmed yet</span></div></div>`;
    this.holdArc = v(2).querySelector(".a");
    this.holdName = v(2).querySelector(".sv-hold-name");
    this.holdLast = v(2).querySelector(".sv-hold-last");
    this.hold = new Spring(0, { k: 220, zeta: 1 });

    // 4 · motion: a speed meter with the trigger thresholds marked.
    v(3).innerHTML = `<div class="sv-swipe"><div class="sv-track"><i class="th l"></i><i class="th r"></i><i class="zero"></i><b class="needle"></b></div><div class="sv-swipe-labels"><span>← swipe left</span><span class="sv-speed">0.0 palm widths</span><span>swipe right →</span></div><div class="sv-flash" aria-hidden="true"></div></div>`;
    this.needle = v(3).querySelector(".needle");
    this.speedText = v(3).querySelector(".sv-speed");
    this.flash = v(3).querySelector(".sv-flash");
    this.speed = new Spring(0, { k: 260, zeta: 0.8 });

    // 5 · private: live counters.
    v(4).innerHTML = `<dl class="sv-private"><div><dt>Uploaded</dt><dd>0 <small>bytes</small></dd></div><div><dt>Frames analysed</dt><dd class="sv-frames">0</dd></div><div><dt>Runs in</dt><dd class="sv-where">–</dd></div></dl>`;
    this.framesEl = v(4).querySelector(".sv-frames");
    this.whereEl = v(4).querySelector(".sv-where");
  }

  setActive(i) {
    this.active = i;
  }

  /** Called on every tracker result. */
  update(state, aspect) {
    this.state = state;
    this.aspect = aspect;
    const hand = state.hands[0];
    // Palm-centre history for the speed meter, in the engine's units.
    if (hand) {
      const d = hand.landmarks;
      const cx = PALM.reduce((a, i) => a + d[i][0], 0) / PALM.length;
      const palm = Math.hypot((d[0][0] - d[9][0]) * aspect, d[0][1] - d[9][1]) * Math.max(aspect, 1);
      const t = performance.now();
      this.history.push({ t, x: cx * aspect, palm });
      while (this.history.length && t - this.history[0].t > 320) this.history.shift();
      const a = this.history[0];
      const b = this.history.at(-1);
      this.speed.set(this.history.length > 1 ? (b.x - a.x) / Math.max((a.palm + b.palm) / 2, 1e-6) : 0);
    } else {
      this.history = [];
      this.speed.set(0);
    }
    for (const ev of state.events) {
      if (ev.type !== "gesture") continue;
      if (ev.id === "swipe_left" || ev.id === "swipe_right") {
        this.flash.textContent = ev.id === "swipe_left" ? "← Swipe left" : "Swipe right →";
        this.flash.classList.remove("on");
        void this.flash.offsetWidth;
        this.flash.classList.add("on");
      } else this.lastEvent = { id: ev.id, t: performance.now() };
    }
  }

  step(dt) {
    const s = this.state;
    const hand = s?.hands[0];
    const color = s?.pair ? "#f5f5f7" : CHANNEL[hand?.handedness] ?? "#f5f5f7";
    switch (this.active) {
      case 0: this.drawLandmarks(dt, hand, color); break;
      case 1: this.drawProbs(dt, hand, s); break;
      case 2: this.drawHold(dt, hand, s, color); break;
      case 3: this.drawSwipe(dt); break;
      case 4: this.drawPrivate(); break;
    }
  }

  drawLandmarks(dt, hand, color) {
    const c = this.lmCanvas;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    if (!hand) {
      this.lmPts = null;
      g.fillStyle = "#6e6e73";
      g.font = "500 13px Inter, sans-serif";
      g.textAlign = "center";
      g.fillText("Raise a hand to see its 21 landmarks", w / 2, h / 2);
      return;
    }
    // Fit the hand into the canvas (keeping its proportions) and ease towards it.
    const k = this.aspect || 1;
    const lm = hand.landmarks.map(([x, y]) => [x * k, y]); // isotropic: undo the camera's aspect ratio
    const xs = lm.map((p) => p[0]);
    const ys = lm.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const scale = Math.min((w - 48) / Math.max(x1 - x0, 1e-3), (h - 40) / Math.max(y1 - y0, 1e-3)) * 0.92;
    const target = lm.map(([x, y]) => [w / 2 + (x - (x0 + x1) / 2) * scale, h / 2 + (y - (y0 + y1) / 2) * scale]);
    if (!this.lmPts) this.lmPts = target.map((p) => [...p]);
    const a = approach(dt, 0.06);
    this.lmPts.forEach((p, i) => { p[0] += (target[i][0] - p[0]) * a; p[1] += (target[i][1] - p[1]) * a; });
    const P = this.lmPts;
    g.lineCap = g.lineJoin = "round";
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.beginPath();
    for (const [i, j] of CONNECTIONS) { g.moveTo(...P[i]); g.lineTo(...P[j]); }
    g.stroke();
    g.font = "500 10px Inter, sans-serif";
    g.textAlign = "left";
    P.forEach(([x, y], i) => {
      g.fillStyle = "#f5f5f7";
      g.beginPath();
      g.arc(x, y, i % 4 === 0 ? 3.5 : 2.5, 0, Math.PI * 2);
      g.fill();
      g.fillStyle = "rgba(245,245,247,0.55)";
      g.fillText(String(i), x + 6, y - 5);
    });
  }

  drawProbs(dt, hand, s) {
    const top = hand ? (s.pair ? [{ id: s.pair.id, p: s.pair.confidence }] : hand.top).slice(0, 3) : [];
    this.probRows.forEach((row, i) => {
      const c = top[i];
      row.li.classList.toggle("empty", !c);
      if (c) {
        const name = this.labelInfo(c.id).name;
        if (row.name.textContent !== name) row.name.textContent = name;
        row.pct.textContent = `${Math.round(c.p * 100)}%`;
      } else if (i === 0) {
        row.name.textContent = "Waiting for a hand";
        row.pct.textContent = "";
      }
      row.s.set(c ? c.p : 0);
      row.bar.style.transform = `scaleX(${Math.max(0, row.s.step(dt)).toFixed(4)})`;
    });
  }

  drawHold(dt, hand, s, color) {
    const id = s?.pair?.id ?? hand?.label;
    const live = id && id !== "no_gesture";
    this.hold.set(live ? s?.pair?.hold ?? hand?.hold ?? 0 : 0);
    this.holdArc.style.strokeDasharray = `${(Math.max(0, Math.min(1, this.hold.step(dt))) * 100).toFixed(2)} 100`;
    this.holdArc.style.stroke = color;
    const name = live ? this.labelInfo(id).name : hand ? "No gesture" : "No hand";
    if (this.holdName.textContent !== name) this.holdName.textContent = name;
    if (this.lastEvent) {
      const ago = Math.round((performance.now() - this.lastEvent.t) / 1000);
      this.holdLast.textContent = `Last confirmed: ${this.labelInfo(this.lastEvent.id).name}, ${ago < 1 ? "just now" : `${ago} s ago`}`;
    }
  }

  drawSwipe(dt) {
    const v = this.speed.step(dt);
    const pos = Math.max(-1, Math.min(1, v / (SWIPE_AT * 1.35)));
    this.needle.style.transform = `translateX(${(pos * 50).toFixed(2)}cqw)`;
    this.needle.classList.toggle("over", Math.abs(v) >= SWIPE_AT);
    this.speedText.textContent = `${Math.abs(v).toFixed(1)} palm widths`;
  }

  drawPrivate() {
    const st = this.stats();
    this.framesEl.textContent = st.frames.toLocaleString();
    this.whereEl.textContent = st.mode === "worker" ? `Worker · ${st.delegate}` : `Page · ${st.delegate ?? "–"}`;
  }
}

export { gestureInfo };
