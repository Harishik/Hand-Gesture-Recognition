import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { CustomGestures } from "./custom.js";
import { DEFAULT_SETTINGS, GestureEngine } from "./engine.js";
import { GESTURES, gestureInfo } from "./gestures-meta.js";
import { glyphURL, loadPoses } from "./glyph.js";
import { GestureModel } from "./mlp.js";
import { HandScene } from "./scene3d.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const CHANNEL = { Left: "#7fa8ff", Right: "#ffb25c" }; // left / right hand, as in the 3D studio
const channelOf = (handedness) => CHANNEL[handedness] ?? "#ece8e1";
const icon = (id, size = 22, color) => `<img class="glyph" src="${glyphURL(id, size, color)}" alt="" width="${size}" height="${size}">`;

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), overlay: $("overlay"), stage: $("stage"), start: $("start"), startBtn: $("start-btn"),
  nowIcon: $("now-icon"), nowName: $("now-name"), nowSub: $("now-sub"), now: $("now"), hold: $("hold-ring"),
  fps: $("fps"), handsView: $("hands-view"), log: $("log"), cursor: $("cursor"), toast: $("toast"), burst: $("burst"),
};
const ctx = els.overlay.getContext("2d");
const lab = new HandScene($("lab3d"));
// Render the 3D lab only while it is on screen, so it doesn't compete with hand tracking for the GPU.
let labVisible = false;
new IntersectionObserver(([entry]) => (labVisible = entry.isIntersecting)).observe($("lab3d"));
(function renderLab() {
  requestAnimationFrame(renderLab);
  if (labVisible) lab.render();
})();

// ---------------------------------------------------------------- actions & the demo deck
const ACTIONS = {
  none: "— none —",
  next_slide: "Next slide",
  prev_slide: "Previous slide",
  play_pause: "Play / pause",
  volume_up: "Volume up",
  volume_down: "Volume down",
  mute: "Mute / unmute",
  like: "Like",
  dislike: "Dislike",
  snapshot: "Take snapshot",
  celebrate: "Celebrate",
  toggle_theme: "Toggle theme",
  toggle_cursor: "Toggle air cursor",
  reset_deck: "Reset deck",
};
const DEFAULT_BINDINGS = {
  swipe_left: "prev_slide", swipe_right: "next_slide", swipe_up: "volume_up", swipe_down: "volume_down",
  palm: "play_pause", like: "like", dislike: "dislike", ok: "snapshot", mute: "mute", rock: "celebrate",
  hand_heart: "celebrate", hand_heart2: "celebrate", take_picture: "snapshot", xsign: "reset_deck",
};
const BINDINGS_KEY = "hgr.bindings.v1";
const SETTINGS_KEY = "hgr.settings.v1";
const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
  },
};
let bindings = { ...DEFAULT_BINDINGS, ...store.get(BINDINGS_KEY, {}) };

const SLIDES = [
  ["Swipe to navigate", "Open your hand and swipe left or right.", "#2b2f36", "#3a3f47"],
  ["33 gestures", "26 one-hand and 7 two-hand gestures from HaGRIDv2.", "#3a3f47", "#2b2f36"],
  ["Air cursor", "Point with your index finger, pinch to click.", "#2b2f36", "#3a3f47"],
  ["Teach it", "Record a custom gesture in 3 seconds.", "#3a3f47", "#2b2f36"],
  ["On-device", "Video never leaves your browser.", "#2b2f36", "#3a3f47"],
];
const deck = { slide: 0, playing: false, volume: 0.6, muted: false, likes: 0, dislikes: 0 };

function renderDeck() {
  const track = $("slide-track");
  if (!track.children.length) {
    track.innerHTML = SLIDES.map(
      ([t, p, a, b]) => `<div class="slide" style="background:linear-gradient(135deg,${a},${b})"><h4>${t}</h4><p>${p}</p></div>`,
    ).join("");
    $("slide-dots").innerHTML = SLIDES.map(() => "<i></i>").join("");
    $("eq").innerHTML = "<span></span>".repeat(24);
  }
  [...track.children].forEach((el, i) => {
    const offset = i - deck.slide;
    el.style.transform = `translateX(${offset * 100}%) scale(${offset === 0 ? 1 : 0.92})`;
    el.style.opacity = offset === 0 ? 1 : 0.4;
  });
  [...$("slide-dots").children].forEach((d, i) => d.classList.toggle("on", i === deck.slide));
  $("play-btn").textContent = deck.playing ? "Pause" : "Play";
  $("media-state").textContent = deck.playing ? (deck.muted ? "Playing · muted" : "Playing") : "Paused";
  $("mute-btn").textContent = deck.muted ? "Unmute" : "Mute";
  $("volume-fill").style.width = `${deck.volume * 100}%`;
  $("volume-label").textContent = `${Math.round(deck.volume * 100)}%`;
  $("likes").textContent = deck.likes;
  $("dislikes").textContent = deck.dislikes;
}

function animateEq(t) {
  const bars = $("eq").children;
  for (let i = 0; i < bars.length; i++) {
    const level = deck.playing && !deck.muted ? (0.25 + 0.75 * Math.abs(Math.sin(t / 180 + i * 0.7) * Math.sin(t / 410 + i))) * deck.volume : 0.06;
    bars[i].style.height = `${Math.max(6, level * 100)}%`;
  }
}

function toast(text) {
  els.toast.textContent = text;
  els.toast.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => els.toast.classList.remove("show"), 1400);
}

function celebrate() {
  for (let i = 0; i < 24; i++) {
    const s = document.createElement("span");
    s.style.background = ["#ece8e1", "#ffb25c", "#7fa8ff"][i % 3];
    s.style.left = `${Math.random() * 100}%`;
    s.style.top = `${70 + Math.random() * 30}%`;
    s.style.animationDelay = `${Math.random() * 0.4}s`;
    els.burst.appendChild(s);
    setTimeout(() => s.remove(), 2200);
  }
}

function snapshot() {
  const c = document.createElement("canvas");
  c.width = 320;
  c.height = Math.round(320 / (els.video.videoWidth / els.video.videoHeight || 16 / 9));
  const g = c.getContext("2d");
  if (settings.mirror) { g.translate(c.width, 0); g.scale(-1, 1); }
  g.drawImage(els.video, 0, 0, c.width, c.height);
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.drawImage(els.overlay, 0, 0, c.width, c.height);
  const strip = $("snap-strip");
  strip.querySelector(".muted")?.remove();
  const img = new Image();
  img.src = c.toDataURL("image/jpeg", 0.85);
  img.alt = "Snapshot";
  strip.prepend(img);
  while (strip.children.length > 8) strip.lastChild.remove();
  els.stage.classList.remove("flash");
  void els.stage.offsetWidth;
  els.stage.classList.add("flash");
}

function runAction(action) {
  const n = SLIDES.length;
  switch (action) {
    case "next_slide": deck.slide = (deck.slide + 1) % n; break;
    case "prev_slide": deck.slide = (deck.slide - 1 + n) % n; break;
    case "play_pause": deck.playing = !deck.playing; break;
    case "volume_up": deck.volume = Math.min(1, deck.volume + 0.1); break;
    case "volume_down": deck.volume = Math.max(0, deck.volume - 0.1); break;
    case "mute": deck.muted = !deck.muted; break;
    case "like": deck.likes++; break;
    case "dislike": deck.dislikes++; break;
    case "snapshot": snapshot(); break;
    case "celebrate": celebrate(); break;
    case "toggle_theme": toggleTheme(); break;
    case "toggle_cursor": settings.cursor = !settings.cursor; syncSettings(); break;
    case "reset_deck": Object.assign(deck, { slide: 0, playing: false, volume: 0.6, muted: false, likes: 0, dislikes: 0 }); $("snap-strip").innerHTML = '<span class="muted">Snapshots cleared</span>'; break;
    default: return false;
  }
  const btn = document.querySelector(`[data-action="${action}"]`);
  if (btn) { btn.classList.remove("pressed"); void btn.offsetWidth; btn.classList.add("pressed"); }
  renderDeck();
  return true;
}

document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-action]");
  if (btn) runAction(btn.dataset.action);
});
$("volume").addEventListener("pointerdown", (e) => setVolumeFromX(e.clientX));
function setVolumeFromX(x) {
  const r = $("volume").getBoundingClientRect();
  deck.volume = Math.min(1, Math.max(0, (x - r.left) / r.width));
  renderDeck();
}

// ---------------------------------------------------------------- theme, tabs, settings
function toggleTheme() {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
  store.set("hgr.theme", root.dataset.theme);
}
if (store.get("hgr.theme", "dark") === "light") document.documentElement.dataset.theme = "light";
$("theme-toggle").addEventListener("click", toggleTheme);

document.querySelectorAll(".tabs button").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    document.querySelectorAll(".tab-body").forEach((p) => (p.hidden = p.dataset.panel !== tab.dataset.tab));
  }),
);

let settings = { skeleton: true };
let engine = null;
const custom = new CustomGestures();

function syncSettings() {
  const map = { "s-conf": "minConfidence", "s-hold": "holdMs", "s-smooth": "smoothing", "s-reject": "rejectBias" };
  for (const [id, key] of Object.entries(map)) {
    $(id).value = settings[key];
    $(id.replace("s-", "o-")).textContent = settings[key];
  }
  for (const [id, key] of Object.entries({ "s-twohand": "twoHand", "s-swipes": "swipes", "s-cursor": "cursor", "s-mirror": "mirror", "s-skeleton": "skeleton" }))
    $(id).checked = settings[key];
  els.video.classList.toggle("mirror", settings.mirror);
  if (engine) Object.assign(engine.settings, settings);
  store.set(SETTINGS_KEY, settings);
}

function bindSettings() {
  for (const [id, key] of Object.entries({ "s-conf": "minConfidence", "s-hold": "holdMs", "s-smooth": "smoothing", "s-reject": "rejectBias" }))
    $(id).addEventListener("input", (e) => { settings[key] = Number(e.target.value); syncSettings(); });
  for (const [id, key] of Object.entries({ "s-twohand": "twoHand", "s-swipes": "swipes", "s-cursor": "cursor", "s-mirror": "mirror", "s-skeleton": "skeleton" }))
    $(id).addEventListener("change", (e) => { settings[key] = e.target.checked; syncSettings(); });
}

function renderBindings() {
  const ids = Object.keys(GESTURES).filter((id) => id !== "no_gesture" && id !== "pinch");
  const all = [...ids, ...custom.items.map((g) => g.id)];
  $("bindings").innerHTML = all
    .map((id) => {
      const info = labelInfo(id);
      const options = Object.entries(ACTIONS).map(([a, label]) => `<option value="${a}" ${bindings[id] === a ? "selected" : ""}>${label}</option>`).join("");
      return `<label class="binding">${icon(id)}${info.name}<select data-gesture="${id}">${options}</select></label>`;
    })
    .join("");
  $("bindings").querySelectorAll("select").forEach((s) =>
    s.addEventListener("change", () => {
      bindings[s.dataset.gesture] = s.value;
      store.set(BINDINGS_KEY, bindings);
    }),
  );
}

function renderGestureGrid() {
  $("gesture-grid").innerHTML = Object.entries(GESTURES)
    .filter(([id]) => id !== "no_gesture")
    .map(([id, g]) => {
      const badge = g.twoHanded ? '<span class="badge">2 hands</span>' : g.motion ? '<span class="badge">motion</span>' : "";
      return `<div class="g-card" data-g="${id}">${icon(id, 40)}<b>${g.name}${badge}</b><small>${g.hint}</small></div>`;
    })
    .join("");
}

function renderCustomList() {
  $("custom-list").innerHTML = custom.items.length
    ? custom.items.map((g) => `<li><span>${icon(g.id, 18)} <b>${g.name}</b> <small class="muted">${g.samples.length} samples</small></span><button class="link" data-remove="${g.id}">Remove</button></li>`).join("")
    : '<li class="muted">No custom gestures yet.</li>';
  $("custom-list").querySelectorAll("[data-remove]").forEach((b) =>
    b.addEventListener("click", () => { custom.remove(b.dataset.remove); renderCustomList(); renderBindings(); }),
  );
}

// ---------------------------------------------------------------- teach (few-shot)
const teach = { phase: "idle", samples: [], until: 0 };
$("teach-btn").addEventListener("click", () => {
  const name = $("teach-name").value.trim();
  if (!name) { $("teach-status").textContent = "Give the gesture a name first."; return; }
  if (!engine) { $("teach-status").textContent = "Start the camera first."; return; }
  teach.phase = "countdown";
  teach.until = performance.now() + 2000;
  teach.samples = [];
  teach.name = name;
});

function updateTeach(state, t) {
  if (teach.phase === "idle") return;
  const status = $("teach-status");
  if (teach.phase === "countdown") {
    status.textContent = `Get ready… ${Math.ceil((teach.until - t) / 1000)}`;
    if (t >= teach.until) { teach.phase = "recording"; teach.until = t + 3000; }
  } else if (teach.phase === "recording") {
    const hand = state.hands[0];
    if (hand) teach.samples.push(engine.hand.embed(hand.features));
    status.textContent = `● Recording ${teach.samples.length} frames… ${((teach.until - t) / 1000).toFixed(1)}s`;
    if (t >= teach.until) {
      teach.phase = "idle";
      if (teach.samples.length < 15) {
        status.textContent = "Not enough frames with a visible hand. Try again.";
      } else {
        const g = custom.add(teach.name, teach.samples);
        status.textContent = `Saved "${g.name}" (${g.samples.length} frames, match threshold ${g.threshold.toFixed(2)}).`;
        renderCustomList();
        renderBindings();
      }
    }
  }
}

// ---------------------------------------------------------------- drawing
function drawHands(state) {
  const { width: w, height: h } = els.overlay;
  ctx.clearRect(0, 0, w, h);
  if (!settings.skeleton) return;
  const scale = w / 640;
  for (const hand of state.hands) {
    const pts = hand.landmarks.map(([x, y]) => [x * w, y * h]);
    ctx.lineCap = "round";
    const color = channelOf(hand.handedness);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3 * scale;
    ctx.beginPath();
    for (const [a, b] of CONNECTIONS) {
      ctx.moveTo(...pts[a]);
      ctx.lineTo(...pts[b]);
    }
    ctx.stroke();
    ctx.fillStyle = "#ece8e1";
    pts.forEach(([x, y], i) => {
      ctx.beginPath();
      ctx.arc(x, y, (i % 4 === 0 ? 4.5 : 3) * scale, 0, Math.PI * 2);
      ctx.fill();
    });
    if (hand.pinch?.active) {
      const [x, y] = [(pts[4][0] + pts[8][0]) / 2, (pts[4][1] + pts[8][1]) / 2];
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2 * scale;
      ctx.beginPath();
      ctx.arc(x, y, 14 * scale, 0, Math.PI * 2);
      ctx.stroke();
    }
    // Label chip above the hand.
    const id = hand.pair?.id ?? hand.label;
    const info = labelInfo(id);
    const minY = Math.min(...pts.map((p) => p[1]));
    const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
    const text = info.name;
    ctx.font = `700 ${16 * scale}px "Bricolage Grotesque", sans-serif`;
    const tw = ctx.measureText(text).width + 18 * scale;
    const y = Math.max(minY - 34 * scale, 6);
    ctx.fillStyle = "rgba(30,33,38,0.9)";
    roundRect(cx - tw / 2, y, tw, 26 * scale, 13 * scale);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.fillText(text, cx - tw / 2 + 9 * scale, y + 18 * scale);
    const hold = hand.pair?.hold ?? hand.hold ?? 0;
    if (id !== "no_gesture" && hold > 0) {
      ctx.fillStyle = color;
      roundRect(cx - tw / 2, y + 23 * scale, tw * hold, 3 * scale, 2 * scale);
      ctx.fill();
    }
  }
  if (state.pair && state.hands.length >= 2) {
    const [a, b] = state.hands;
    ctx.setLineDash([6 * scale, 6 * scale]);
    ctx.strokeStyle = "rgba(236,232,225,0.7)";
    ctx.lineWidth = 2 * scale;
    ctx.beginPath();
    ctx.moveTo(a.landmarks[0][0] * w, a.landmarks[0][1] * h);
    ctx.lineTo(b.landmarks[0][0] * w, b.landmarks[0][1] * h);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
}

function labelInfo(id) {
  if (id?.startsWith("custom:")) {
    const g = custom.items.find((c) => c.id === id);
    return { name: g ? g.name : id };
  }
  return gestureInfo(id);
}

// ---------------------------------------------------------------- panels
const perf = { fps: 0, land: 0, cls: 0, history: [] };
let lastPanel = 0;

function updatePanels(state, t) {
  const primary = state.pair ? null : state.hands[0];
  const id = state.pair?.id ?? primary?.label ?? null;
  const info = id ? labelInfo(id) : { name: "Show your hand" };
  els.nowIcon.innerHTML = id && id !== "no_gesture" ? icon(id, 40, channelOf(primary?.handedness)) : "";
  els.nowName.textContent = state.hands.length ? info.name : "Show your hand";
  const conf = state.pair?.confidence ?? primary?.confidence;
  els.nowSub.textContent = state.hands.length
    ? `${state.pair ? "two-hand" : primary.handedness.toLowerCase() + " hand"} · ${Math.round((conf ?? 0) * 100)}%`
    : "No hands in view";
  const hold = state.pair?.hold ?? primary?.hold ?? 0;
  els.hold.style.strokeDashoffset = String(94.25 * (1 - (id === "no_gesture" ? 0 : hold)));

  if (t - lastPanel < 100) return;
  lastPanel = t;
  els.fps.textContent = `${perf.fps.toFixed(0)} FPS`;
  $("m-fps").textContent = perf.fps.toFixed(0);
  $("m-land").textContent = `${perf.land.toFixed(1)} ms`;
  $("m-cls").textContent = `${perf.cls.toFixed(2)} ms`;
  $("m-hands").textContent = state.hands.length;
  drawSparkline();

  const live = new Set(state.hands.map((h) => h.pair?.id ?? h.label));
  document.querySelectorAll(".g-card").forEach((c) => c.classList.toggle("live", live.has(c.dataset.g)));

  if (!state.hands.length) {
    els.handsView.innerHTML = '<p class="muted">No hands in view.</p>';
    return;
  }
  const pair = state.pair
    ? `<div class="pair-banner">${icon(state.pair.id, 22)} ${labelInfo(state.pair.id).name}, two-hand, ${Math.round(state.pair.confidence * 100)}%</div>`
    : "";
  els.handsView.innerHTML =
    pair +
    state.hands
      .map((h) => {
        const bars = h.top
          .map((c) => {
            const g = gestureInfo(c.id);
            return `<div class="bar">${icon(c.id, 20, "#8e939c")}<div><div class="name">${g.name}</div><div class="track"><div class="fill" style="width:${(c.p * 100).toFixed(1)}%"></div></div></div><span class="pct">${(c.p * 100).toFixed(0)}%</span></div>`;
          })
          .join("");
        const customLine = h.custom ? `<div class="bar"><span>✦</span><div><div class="name">custom: ${h.custom.name}</div></div><span class="pct">${(h.custom.similarity * 100).toFixed(0)}%</span></div>` : "";
        return `<div class="hand-card"><header><b>${labelInfo(h.label).name}</b><span>${h.handedness} hand${h.pinch?.active ? " · pinching" : ""}</span></header>${customLine}${bars}</div>`;
      })
      .join("");
}

function drawSparkline() {
  const c = $("sparkline");
  const w = (c.width = c.clientWidth * devicePixelRatio);
  const h = (c.height = 48 * devicePixelRatio);
  const g = c.getContext("2d");
  const data = perf.history;
  if (data.length < 2) return;
  const max = Math.max(60, ...data);
  g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent");
  g.lineWidth = 2 * devicePixelRatio;
  g.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - (v / max) * (h - 4) - 2;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
}

function logEvent(ev, action) {
  const info = labelInfo(ev.id);
  const li = document.createElement("li");
  const time = new Date().toLocaleTimeString([], { hour12: false });
  li.innerHTML = `${icon(ev.id, 20)}<span>${info.name}${action ? `<br><span class="act">${ACTIONS[action]}</span>` : ""}</span><time>${time}</time>`;
  els.log.prepend(li);
  while (els.log.children.length > 30) els.log.lastChild.remove();
}

// ---------------------------------------------------------------- air cursor
let dragTarget = null;
function updateCursor(state) {
  const c = state.cursor;
  if (!c) { els.cursor.style.display = "none"; dragTarget = null; return; }
  const x = c.x * innerWidth;
  const y = c.y * innerHeight;
  els.cursor.style.display = "block";
  els.cursor.style.transform = `translate(${x}px, ${y}px)`;
  els.cursor.classList.toggle("pinch", c.pinching);
  for (const ev of state.events) {
    if (ev.type === "pinch_start") {
      const el = document.elementFromPoint(x, y);
      const drag = el?.closest("[data-pinch-drag]");
      if (drag) dragTarget = drag;
      else el?.closest("button, [data-action], select, label")?.click();
      logEvent({ id: "pinch" }, null);
    }
    if (ev.type === "pinch_end") dragTarget = null;
  }
  if (dragTarget && c.pinching) setVolumeFromX(x);
}

// ---------------------------------------------------------------- main loop
let landmarker = null;
let lastVideoTime = -1;

async function loadModels() {
  const status = $("status-model");
  status.className = "chip busy";
  try {
    const [hand, pair, vision] = await Promise.all([
      GestureModel.load("models/hand.json"),
      GestureModel.load("models/pair.json"),
      FilesetResolver.forVisionTasks(WASM),
    ]);
    let delegate = "GPU";
    try {
      landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: LANDMARKER_MODEL, delegate },
        runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: 0.5,
      });
    } catch {
      delegate = "CPU";
      landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: LANDMARKER_MODEL, delegate },
        runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: 0.5,
      });
    }
    engine = new GestureEngine(hand, pair, custom);
    Object.assign(engine.settings, settings);
    status.className = "chip ok";
    status.lastChild.textContent = `${hand.classes.length - 1} + ${pair.classes.length - 1} gestures`;
    const d = $("status-delegate");
    d.className = "chip ok";
    d.lastChild.textContent = delegate;
    els.startBtn.disabled = false;
    els.startBtn.textContent = "Start camera";
    if (new URLSearchParams(location.search).has("autostart")) startCamera(); // demos and screenshots
  } catch (err) {
    status.className = "chip err";
    els.startBtn.textContent = "Failed to load models";
    console.error(err);
  }
}

async function startCamera() {
  const cam = $("status-camera");
  cam.className = "chip busy";
  const demo = new URLSearchParams(location.search).get("video");
  try {
    if (demo) {
      els.video.src = demo;
      els.video.loop = true;
    } else {
      els.video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
    }
    await els.video.play();
    await new Promise((r) => (els.video.readyState >= 2 ? r() : els.video.addEventListener("loadeddata", r, { once: true })));
    const { videoWidth: vw, videoHeight: vh } = els.video;
    els.overlay.width = vw;
    els.overlay.height = vh;
    els.stage.style.aspectRatio = `${vw} / ${vh}`;
    els.start.hidden = true;
    cam.className = "chip ok";
    cam.lastChild.textContent = demo ? "Demo video" : `${vw}×${vh}`;
    requestAnimationFrame(loop);
  } catch (err) {
    cam.className = "chip err";
    cam.lastChild.textContent = "Camera blocked";
    els.startBtn.textContent = "Allow camera access and retry";
    console.error(err);
  }
}

let lastFrameT = performance.now();
function loop(t) {
  requestAnimationFrame(loop);
  animateEq(t);
  if (els.video.currentTime === lastVideoTime) return;
  lastVideoTime = els.video.currentTime;

  const t0 = performance.now();
  const result = landmarker.detectForVideo(els.video, t0);
  const landMs = performance.now() - t0;
  const handedness = result.handedness ?? result.handednesses ?? [];
  const hands = result.landmarks.map((lm, i) => ({
    landmarks: lm.map((p) => [p.x, p.y]),
    handedness: handedness[i]?.[0]?.categoryName ?? "Hand",
  }));
  const state = engine.update(hands, t0, els.video.videoWidth / els.video.videoHeight);

  const dt = t0 - lastFrameT;
  lastFrameT = t0;
  perf.fps = perf.fps ? 0.9 * perf.fps + 0.1 * (1000 / Math.max(dt, 1)) : 1000 / Math.max(dt, 1);
  perf.land = perf.land ? 0.9 * perf.land + 0.1 * landMs : landMs;
  perf.cls = perf.cls ? 0.9 * perf.cls + 0.1 * state.classifyMs : state.classifyMs;
  perf.history.push(perf.fps);
  if (perf.history.length > 120) perf.history.shift();

  for (const ev of state.events) {
    if (ev.type !== "gesture") continue;
    const action = bindings[ev.id];
    const ran = action && action !== "none" && runAction(action);
    if (ran) toast(`${labelInfo(ev.id).name}: ${ACTIONS[action]}`);
    logEvent(ev, ran ? action : null);
    els.now.classList.remove("fired");
    void els.now.offsetWidth;
    els.now.classList.add("fired");
  }

  try {
    lab.update(
      state.hands.map((_, i) => result.worldLandmarks?.[i]?.map((p) => [p.x, p.y, p.z])),
      state.hands.map((h) => h.landmarks),
      state.hands.map((h) => h.pair?.id ?? h.label),
      state.events,
      settings.mirror,
    );
  } catch (err) {
    console.error("3D lab update failed", err); // never let the 3D view stop recognition
  }
  drawHands(state);
  updatePanels(state, t0);
  updateCursor(state);
  updateTeach(state, t0);
  window.__gestureStudio = { state, perf: { ...perf, history: undefined } }; // for automated checks
}

// ---------------------------------------------------------------- boot
settings = { ...settings, ...DEFAULT_SETTINGS, ...store.get(SETTINGS_KEY, {}) };
bindSettings();
syncSettings();
await loadPoses();
document.querySelectorAll("img[data-glyph]").forEach((img) => (img.src = glyphURL(img.dataset.glyph, 26)));
renderDeck();
renderGestureGrid();
renderBindings();
renderCustomList();
els.startBtn.addEventListener("click", startCamera);
loadModels();
