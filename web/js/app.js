// Gesture Studio Classic: bind gestures to actions, teach your own, tune recognition, and drive a demo deck.
// Same motion model as the 3D studio: tracker frames (~30 Hz) set targets, the display loop eases towards them.
import { DEFAULT_SETTINGS } from "./engine.js";
import { GESTURES, gestureInfo } from "./gestures-meta.js";
import { glyphURL, loadPoses } from "./glyph.js";
import { PoseHero } from "./hero.js";
import { approach, onFrame, reducedMotion, Spring, SpringList, stableRank, TweenNumber } from "./motion.js";
import { HandScene } from "./scene3d.js";
import { describeCameraError, HandTracker } from "./tracker.js";
import { announce, bindHelp, notice, session, typing } from "./ui.js";

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const CHANNEL = { Left: "#7fa8ff", Right: "#ffb25c" }; // left / right hand, as in the 3D studio
const channelOf = (handedness) => CHANNEL[handedness] ?? "#ece8e1";
const icon = (id, size = 22, color) => `<img class="glyph" src="${glyphURL(id, size, color)}" alt="" width="${size}" height="${size}">`;
const DEMO_CLIP = "test-media/demo.webm";
const params = new URLSearchParams(location.search);

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), overlay: $("overlay"), stage: $("stage"), start: $("start"), startBtn: $("start-btn"),
  nowIcon: $("now-icon"), nowName: $("now-name"), nowSub: $("now-sub"), now: $("now"), hold: $("hold-ring"),
  log: $("log"), cursor: $("cursor"), burst: $("burst"), coach: $("coach"),
};
const ctx = els.overlay.getContext("2d");
const tracker = new HandTracker(els.video);
const custom = tracker.custom;
let lab = null;
try {
  lab = new HandScene($("lab3d"));
} catch (err) {
  console.warn("3D lab unavailable (no WebGL)", err);
  document.querySelector(".card.lab").hidden = true;
}
// Render the 3D lab only while it is on screen, so it doesn't compete with hand tracking for the GPU.
let labVisible = false;
new IntersectionObserver(([entry]) => (labVisible = entry.isIntersecting)).observe($("lab3d"));
onFrame((dt) => {
  if (!labVisible || !lab) return;
  lab.tau = tracker.smoothTau;
  lab.render(dt);
});

// ---------------------------------------------------------------- actions & the demo deck
const ACTIONS = {
  none: "No action",
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
const COACH_KEY = "hgr.coach.cursor";
const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
  },
};
let bindings = { ...DEFAULT_BINDINGS, ...store.get(BINDINGS_KEY, {}) };

// Each slide: kicker, title, body, stat, glyphs drawn from the real pose data, accent colour.
const SLIDES = [
  ["Navigate", "Swipe to move", "Open your hand and flick left or right. Speed is normalised by palm size, so it works at any distance.", "320 ms", ["palm", "stop", "palm"], "#7fa8ff"],
  ["Vocabulary", "33 gestures", "26 one-hand and 7 two-hand gestures, trained on 1.3M hands from HaGRIDv2.", "98.8%", ["peace", "ok", "rock", "like", "call", "fist"], "#ece8e1"],
  ["Pointer", "Air cursor", "Your index fingertip, smoothed by a One Euro filter. Pinch to click, pinch and drag sliders.", "~1 ms", ["point", "grip", "one"], "#ffb25c"],
  ["Personalise", "Teach it yours", "Record any pose for three seconds. It is matched in the model's learned embedding.", "3 s", ["three_gun", "little_finger", "thumb_index"], "#7fa8ff"],
  ["Privacy", "Never uploaded", "Tracking and classification run entirely in this tab, on your GPU. No frames leave the device.", "0 bytes", ["holy", "hand_heart"], "#6fd39b"],
];
const deck = { slide: 0, playing: false, volume: 0.6, muted: false, likes: 0, dislikes: 0, snaps: 0 };
const deckMotion = { slide: new Spring(0, { k: 120, zeta: 0.86 }), volume: new Spring(0.6, { k: 260, zeta: 1 }) };

function renderDeck() {
  const track = $("slide-track");
  if (!track.children.length) {
    track.innerHTML = SLIDES.map(
      ([k, t, p, stat, glyphs, c], i) => `<div class="slide" style="--c:${c}" role="group" aria-roledescription="slide" aria-label="${i + 1} of ${SLIDES.length}">
        <div class="slide-art">${glyphs.map((g, j) => `<img src="${glyphURL(g, 64, c)}" alt="" style="--j:${j}">`).join("")}</div>
        <div class="slide-copy"><span class="slide-kicker">${k}</span><h4>${t}</h4><p>${p}</p></div>
        <b class="slide-stat">${stat}</b></div>`,
    ).join("");
    $("slide-dots").innerHTML = SLIDES.map(() => "<i></i>").join("");
    $("eq").innerHTML = "<span></span>".repeat(28);
  }
  [...track.children].forEach((el, i) => el.setAttribute("aria-hidden", String(i !== deck.slide)));
  [...$("slide-dots").children].forEach((d, i) => d.classList.toggle("on", i === deck.slide));
  deckMotion.slide.set(deck.slide);
  deckMotion.volume.set(deck.volume);
  $("play-btn").setAttribute("aria-label", deck.playing ? "Pause" : "Play");
  $("play-icon").setAttribute("d", deck.playing ? "M4 2.5h3v11H4zM9 2.5h3v11H9z" : "M4 2.5v11l9-5.5z");
  $("media-state").textContent = deck.playing ? (deck.muted ? "Playing · muted" : "Playing") : "Paused";
  $("mute-btn").textContent = deck.muted ? "Unmute" : "Mute";
  $("volume-label").textContent = `${Math.round(deck.volume * 100)}%`;
  $("volume").setAttribute("aria-valuenow", Math.round(deck.volume * 100));
  $("likes").textContent = deck.likes;
  $("dislikes").textContent = deck.dislikes;
}

// Slides, volume and the equaliser move every display frame.
const eqLevels = new Float32Array(28);
onFrame((dt, now) => {
  const pos = deckMotion.slide.step(dt);
  const slides = $("slide-track").children;
  for (let i = 0; i < slides.length; i++) {
    const offset = i - pos;
    const d = Math.min(1, Math.abs(offset));
    slides[i].style.transform = `translate3d(${(offset * 100).toFixed(3)}%, 0, 0) scale(${(1 - 0.08 * d).toFixed(4)})`;
    slides[i].style.opacity = (1 - 0.65 * d).toFixed(3);
  }
  const v = Math.max(0, Math.min(1, deckMotion.volume.step(dt)));
  $("volume-fill").style.transform = `scaleX(${v.toFixed(4)})`;
  $("volume-knob").style.left = `${(v * 100).toFixed(2)}%`;
  const bars = $("eq").children;
  const playing = deck.playing && !deck.muted;
  for (let i = 0; i < bars.length; i++) {
    const target = playing && !reducedMotion.matches
      ? (0.25 + 0.75 * Math.abs(Math.sin(now / 180 + i * 0.7) * Math.sin(now / 410 + i))) * (0.3 + 0.7 * deck.volume)
      : 0.06;
    eqLevels[i] += (target - eqLevels[i]) * approach(dt, 0.07);
    bars[i].style.transform = `scaleY(${Math.max(0.06, eqLevels[i]).toFixed(3)})`;
  }
});

function celebrate() {
  if (reducedMotion.matches) return;
  for (let i = 0; i < 28; i++) {
    const s = document.createElement("span");
    s.style.background = ["#ece8e1", "#ffb25c", "#7fa8ff"][i % 3];
    s.style.left = `${Math.random() * 100}%`;
    s.style.top = `${70 + Math.random() * 30}%`;
    s.style.animationDelay = `${Math.random() * 0.4}s`;
    els.burst.appendChild(s);
    setTimeout(() => s.remove(), 2200);
  }
}

/**
 * Shows only the picture's content box (letterbox bars cropped), centred on the dark stage. The stage keeps a
 * sensible shape (4:3 to 16:9, never taller than wide); the video and overlay are positioned so the content box fills
 * its slot, and clip-path hides the bars. Honours the mirror setting.
 */
function applyCrop() {
  const { videoWidth: vw, videoHeight: vh } = els.video;
  if (!vw) return;
  const c = tracker.crop ?? { x: 0, y: 0, w: 1, h: 1 };
  const content = (vw * c.w) / (vh * c.h);
  const stage = Math.min(16 / 9, Math.max(4 / 3, content));
  els.stage.style.aspectRatio = String(stage);
  // The content box's slot on the stage, in % of the stage.
  const slotW = content < stage ? (content / stage) * 100 : 100;
  const slotH = content > stage ? (stage / content) * 100 : 100;
  const slotL = (100 - slotW) / 2;
  const slotT = (100 - slotH) / 2;
  const x0 = settings.mirror ? 1 - c.x - c.w : c.x; // content box in display (mirrored) coordinates
  const pct = (v) => `${v.toFixed(3)}%`;
  for (const el of [els.video, els.overlay]) {
    // The video element is mirrored by its own transform, so its clip uses source coordinates; the overlay is
    // drawn in display coordinates.
    const cx = el === els.video ? c.x : x0;
    Object.assign(el.style, {
      inset: "auto",
      width: pct(slotW / c.w),
      height: pct(slotH / c.h),
      left: pct(slotL - (x0 / c.w) * slotW),
      top: pct(slotT - (c.y / c.h) * slotH),
      clipPath: `inset(${pct(c.y * 100)} ${pct((1 - cx - c.w) * 100)} ${pct((1 - c.y - c.h) * 100)} ${pct(cx * 100)} round 2px)`,
    });
  }
}

function snapshot() {
  if (!running) {
    notice("Start the camera first, then snap.", { action: { label: "Start camera", run: () => begin() } });
    return false;
  }
  const c = document.createElement("canvas");
  c.width = 320;
  const { videoWidth: vw, videoHeight: vh } = els.video;
  const k = tracker.crop ?? { x: 0, y: 0, w: 1, h: 1 };
  c.height = Math.round((320 * vh * k.h) / (vw * k.w)) || 180;
  const g = c.getContext("2d");
  // Snapshot exactly what the stage shows: the content box, mirrored if the view is.
  const x0 = settings.mirror ? 1 - k.x - k.w : k.x;
  if (settings.mirror) { g.translate(c.width, 0); g.scale(-1, 1); }
  g.drawImage(els.video, k.x * vw, k.y * vh, k.w * vw, k.h * vh, 0, 0, c.width, c.height);
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.drawImage(els.overlay, x0 * vw, k.y * vh, k.w * vw, k.h * vh, 0, 0, c.width, c.height);
  const strip = $("snap-strip");
  const img = new Image();
  img.src = c.toDataURL("image/jpeg", 0.85);
  img.alt = `Snapshot ${++deck.snaps}`;
  strip.prepend(img);
  while (strip.children.length > 8) strip.lastChild.remove();
  syncSnaps();
  els.stage.classList.remove("flash");
  void els.stage.offsetWidth;
  els.stage.classList.add("flash");
  return true;
}
function syncSnaps() {
  const n = $("snap-strip").children.length;
  $("snap-empty").hidden = n > 0;
  $("snap-count").textContent = n ? String(n) : "";
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
    case "snapshot": if (!snapshot()) return false; break;
    case "celebrate": celebrate(); break;
    case "toggle_cursor": settings.cursor = !settings.cursor; syncSettings(); break;
    case "reset_deck":
      Object.assign(deck, { slide: 0, playing: false, volume: 0.6, muted: false, likes: 0, dislikes: 0 });
      $("snap-strip").replaceChildren();
      syncSnaps();
      break;
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
  const go = e.target.closest("[data-goto]");
  if (go) selectTab(go.dataset.goto, true);
});

// Volume: pointer drag, keyboard (it is a proper slider), and pinch-drag via the air cursor.
const volume = $("volume");
function setVolumeFromX(x) {
  const r = volume.getBoundingClientRect();
  deck.volume = Math.min(1, Math.max(0, (x - r.left) / r.width));
  renderDeck();
}
volume.addEventListener("pointerdown", (e) => {
  volume.setPointerCapture(e.pointerId);
  volume.classList.add("dragging");
  setVolumeFromX(e.clientX);
});
volume.addEventListener("pointermove", (e) => volume.hasPointerCapture(e.pointerId) && setVolumeFromX(e.clientX));
volume.addEventListener("pointerup", () => volume.classList.remove("dragging"));
volume.addEventListener("keydown", (e) => {
  const step = { ArrowRight: 0.05, ArrowUp: 0.05, ArrowLeft: -0.05, ArrowDown: -0.05, PageUp: 0.2, PageDown: -0.2 }[e.key];
  if (e.key === "Home" || e.key === "End") deck.volume = e.key === "Home" ? 0 : 1;
  else if (step) deck.volume = Math.min(1, Math.max(0, deck.volume + step));
  else return;
  e.preventDefault();
  e.stopPropagation();
  renderDeck();
});

// ---------------------------------------------------------------- tabs
const tabs = [...document.querySelectorAll('[role="tab"]')];
let currentTab = "live";
function placeInk() {
  const tab = tabs.find((t) => t.dataset.tab === currentTab);
  const ink = $("tab-ink");
  ink.style.width = `${tab.offsetWidth}px`;
  ink.style.transform = `translateX(${tab.offsetLeft}px)`;
}
function selectTab(name, focus = false) {
  if (name === currentTab) return;
  const from = tabs.findIndex((t) => t.dataset.tab === currentTab);
  const to = tabs.findIndex((t) => t.dataset.tab === name);
  currentTab = name;
  tabs.forEach((t) => {
    const on = t.dataset.tab === name;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll(".tab-body").forEach((p) => {
    const on = p.dataset.panel === name;
    p.hidden = !on;
    if (on) {
      p.style.setProperty("--dir", to > from ? 1 : -1);
      p.classList.remove("entering");
      void p.offsetWidth;
      p.classList.add("entering");
    }
  });
  placeInk();
  if (focus) tabs[to].focus();
  tabs[to].scrollIntoView({ block: "nearest", inline: "nearest" });
}
tabs.forEach((tab, i) => {
  tab.addEventListener("click", () => selectTab(tab.dataset.tab));
  tab.addEventListener("keydown", (e) => {
    const move = { ArrowRight: 1, ArrowLeft: -1, Home: -i, End: tabs.length - 1 - i }[e.key];
    if (move === undefined) return;
    e.preventDefault();
    selectTab(tabs[(i + move + tabs.length) % tabs.length].dataset.tab, true);
  });
});
new ResizeObserver(placeInk).observe(document.querySelector(".tabs"));
document.fonts.ready.then(placeInk);

// ---------------------------------------------------------------- settings
let settings = { skeleton: true };
const RANGES = { "s-conf": "minConfidence", "s-hold": "holdMs", "s-smooth": "smoothing", "s-reject": "rejectBias" };
const TOGGLES = { "s-twohand": "twoHand", "s-swipes": "swipes", "s-cursor": "cursor", "s-mirror": "mirror", "s-skeleton": "skeleton" };
const FORMAT = {
  minConfidence: (v) => `${Math.round(v * 100)}%`,
  holdMs: (v) => `${v} ms`,
  smoothing: (v) => v.toFixed(2),
  rejectBias: (v) => (v > 0 ? `+${v}` : `${v}`),
};

function syncSettings() {
  for (const [id, key] of Object.entries(RANGES)) {
    $(id).value = settings[key];
    $(id.replace("s-", "o-")).textContent = FORMAT[key](Number(settings[key]));
  }
  for (const [id, key] of Object.entries(TOGGLES)) $(id).checked = settings[key];
  els.video.classList.toggle("mirror", settings.mirror);
  if (running) applyCrop();
  $("cursor-toggle").setAttribute("aria-pressed", String(settings.cursor));
  if (tracker.engine) Object.assign(tracker.engine.settings, settings);
  store.set(SETTINGS_KEY, settings);
}

function bindSettings() {
  for (const [id, key] of Object.entries(RANGES))
    $(id).addEventListener("input", (e) => { settings[key] = Number(e.target.value); syncSettings(); });
  for (const [id, key] of Object.entries(TOGGLES))
    $(id).addEventListener("change", (e) => { settings[key] = e.target.checked; syncSettings(); });
  $("cursor-toggle").addEventListener("click", () => {
    settings.cursor = !settings.cursor;
    syncSettings();
    notice(settings.cursor ? "Air cursor on: your index fingertip points, a pinch clicks." : "Air cursor off.", { tone: settings.cursor ? "live" : "busy", timeout: 2500 });
  });
}

function actionOptions(selected) {
  return Object.entries(ACTIONS).map(([a, label]) => `<option value="${a}" ${selected === a ? "selected" : ""}>${label}</option>`).join("");
}

function renderBindings() {
  const ids = Object.keys(GESTURES).filter((id) => id !== "no_gesture" && id !== "pinch");
  const all = [...custom.items.map((g) => g.id), ...ids];
  $("bindings").innerHTML = all
    .map((id) => {
      const bound = bindings[id] && bindings[id] !== "none";
      return `<label class="binding">${icon(id)}<span>${labelInfo(id).name}</span><select data-gesture="${id}" class="${bound ? "" : "unbound"}">${actionOptions(bindings[id] ?? "none")}</select></label>`;
    })
    .join("");
  $("bindings").querySelectorAll("select").forEach((s) =>
    s.addEventListener("change", () => {
      bindings[s.dataset.gesture] = s.value;
      s.classList.toggle("unbound", s.value === "none");
      store.set(BINDINGS_KEY, bindings);
    }),
  );
}
$("reset-bindings").addEventListener("click", () => {
  const before = bindings;
  bindings = { ...DEFAULT_BINDINGS };
  store.set(BINDINGS_KEY, bindings);
  renderBindings();
  notice("Bindings restored to the defaults.", { tone: "live", action: { label: "Undo", run: () => { bindings = before; store.set(BINDINGS_KEY, bindings); renderBindings(); } } });
});

function renderGestureGrid() {
  $("gesture-grid").innerHTML = Object.entries(GESTURES)
    .filter(([id]) => id !== "no_gesture")
    .map(([id, g]) => {
      const badge = g.twoHanded ? '<span class="badge">2 hands</span>' : g.motion ? '<span class="badge">motion</span>' : "";
      return `<div class="g-card" data-g="${id}">${icon(id, 36)}<b>${g.name}${badge}</b><small>${g.hint}</small></div>`;
    })
    .join("");
}

function renderCustomList() {
  $("custom-count").textContent = custom.items.length ? String(custom.items.length) : "";
  $("custom-list").innerHTML = custom.items.length
    ? custom.items.map((g) => `<li data-g="${g.id}">${icon(g.id, 18)}<span><b>${g.name}</b><br><small>${g.samples.length} frames · match ≥ ${g.threshold.toFixed(2)}</small></span><button class="btn ghost small" data-remove="${g.id}">Remove</button></li>`).join("")
    : '<li class="empty">Nothing taught yet. Your first gesture will appear here.</li>';
  $("custom-list").querySelectorAll("[data-remove]").forEach((b) =>
    b.addEventListener("click", () => {
      const g = custom.items.find((c) => c.id === b.dataset.remove);
      custom.remove(g.id);
      renderCustomList();
      renderBindings();
      notice(`Removed "${g.name}".`, { action: { label: "Undo", run: () => { custom.add(g.name, g.samples); renderCustomList(); renderBindings(); } } });
    }),
  );
}

// ---------------------------------------------------------------- teach (few-shot)
const teach = { phase: "idle", samples: [], until: 0, start: 0 };
const teachName = $("teach-name");
const teachStatus = $("teach-status");
function setTeachStatus(text, tone = "") {
  teachStatus.textContent = text;
  teachStatus.className = `teach-status ${tone}`;
}
function setStep(n) {
  document.querySelectorAll(".steps li").forEach((li) => {
    const s = Number(li.dataset.step);
    li.classList.toggle("on", s === n);
    li.classList.toggle("done", s < n);
  });
}
teachName.addEventListener("input", () => {
  const name = teachName.value.trim();
  $("teach-btn").disabled = !name;
  const exists = custom.items.some((g) => g.name.toLowerCase() === name.toLowerCase());
  if (exists) setTeachStatus(`"${name}" exists already. Recording again replaces it.`, "warn");
  else if (teach.phase === "idle") setTeachStatus("After a 2-second countdown, hold the pose for 3 seconds and move it slightly, so it's recognised from different angles.");
});
$("teach-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const name = teachName.value.trim();
  if (!name) { teachName.focus(); return; }
  if (!running) {
    setTeachStatus("The camera needs to be on to record.", "warn");
    notice("Turn on the camera to record a gesture.", { action: { label: "Start camera", run: () => begin() } });
    return;
  }
  $("teach-done").hidden = true;
  teach.phase = "countdown";
  teach.start = performance.now();
  teach.until = teach.start + 2000;
  teach.samples = [];
  teach.name = name;
  $("teach-stage").hidden = false;
  $("teach-stage").classList.remove("rec");
  $("teach-btn").disabled = true;
  setStep(2);
  setTeachStatus("Get your hand in view…");
  if (els.stage.getBoundingClientRect().top < 0) els.stage.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "start" });
});
function cancelTeach(reason = "Recording cancelled.") {
  if (teach.phase === "idle") return;
  teach.phase = "idle";
  $("teach-stage").hidden = true;
  $("teach-btn").disabled = !teachName.value.trim();
  setStep(1);
  setTeachStatus(reason, "warn");
}
$("teach-cancel").addEventListener("click", () => cancelTeach());
const teachArc = new Spring(0, { k: 300, zeta: 1 });
onFrame((dt) => {
  $("teach-arc").style.strokeDasharray = `${(Math.max(0, teachArc.step(dt)) * 100).toFixed(2)} 100`;
});

function updateTeach(state, t) {
  if (teach.phase === "idle") return;
  const big = $("teach-big");
  const small = $("teach-small");
  if (teach.phase === "countdown") {
    big.textContent = Math.ceil((teach.until - t) / 1000);
    small.textContent = state.hands.length ? "Hold the pose" : "Show your hand";
    teachArc.set((t - teach.start) / 2000);
    if (t >= teach.until) {
      teach.phase = "recording";
      teach.start = t;
      teach.until = t + 3000;
      teachArc.jump(0);
      $("teach-stage").classList.add("rec");
    }
  } else if (teach.phase === "recording") {
    const hand = state.hands[0];
    if (hand) teach.samples.push(tracker.engine.hand.embed(hand.features));
    big.textContent = teach.samples.length;
    small.textContent = hand ? "Recording · move slightly" : "No hand · keep it in view";
    teachArc.set(Math.min(1, (t - teach.start) / 3000));
    setTeachStatus(`Recording ${teach.samples.length} frames…`);
    // Slow devices (CPU fallback) deliver few frames: keep recording up to 6 s until 15 hand frames are in.
    const enough = teach.samples.length >= 15 || t - teach.start >= 6000;
    if (t >= teach.until && enough) {
      teach.phase = "idle";
      $("teach-stage").hidden = true;
      $("teach-btn").disabled = false;
      if (teach.samples.length < 15) {
        setStep(1);
        setTeachStatus(`Only ${teach.samples.length} frames had a visible hand (15 needed). Keep your hand inside the frame and try again.`, "warn");
      } else {
        const g = custom.add(teach.name, teach.samples);
        setStep(3);
        setTeachStatus(`Saved from ${g.samples.length} frames. Make the pose now: it should light up below.`, "ok");
        $("done-name").textContent = g.name;
        const sel = $("done-bind");
        sel.innerHTML = actionOptions(bindings[g.id] ?? "none");
        sel.dataset.gesture = g.id;
        $("teach-done").hidden = false;
        teachName.value = "";
        renderCustomList();
        renderBindings();
      }
    }
  }
}
$("done-bind").addEventListener("change", (e) => {
  bindings[e.target.dataset.gesture] = e.target.value;
  store.set(BINDINGS_KEY, bindings);
  renderBindings();
  notice(`Bound to "${ACTIONS[e.target.value]}".`, { tone: "live", timeout: 2500 });
});

// ---------------------------------------------------------------- stage overlay (interpolated every display frame)
const drawn = new Map(); // hand key -> { pts, target, alpha, present, hand }
let latest = null;

function trackOverlay(state) {
  const w = els.overlay.width;
  const h = els.overlay.height;
  const seen = new Set();
  for (const hand of state.hands) {
    seen.add(hand.key);
    let d = drawn.get(hand.key);
    const target = hand.landmarks.map(([x, y]) => [x * w, y * h]);
    if (!d) {
      d = { pts: target.map((p) => [...p]), alpha: 0 };
      drawn.set(hand.key, d);
    }
    Object.assign(d, { target, present: true, hand, hold: hand.pair?.hold ?? hand.hold ?? 0 });
  }
  for (const [key, d] of drawn) if (!seen.has(key)) d.present = false;
}

function drawOverlay(dt) {
  const { width: w, height: h } = els.overlay;
  ctx.clearRect(0, 0, w, h);
  if (!running) return;
  const a = approach(dt, tracker.smoothTau);
  const scale = w / 640;
  for (const [key, d] of drawn) {
    d.alpha += ((d.present ? 1 : 0) - d.alpha) * approach(dt, d.present ? 0.08 : 0.15);
    if (!d.present && d.alpha < 0.01) { drawn.delete(key); continue; }
    d.pts.forEach((p, i) => { p[0] += (d.target[i][0] - p[0]) * a; p[1] += (d.target[i][1] - p[1]) * a; });
    d.holdShown = (d.holdShown ?? 0) + (d.hold - (d.holdShown ?? 0)) * approach(dt, 0.06);
    if (!settings.skeleton) continue;
    const { pts, hand } = d;
    const color = channelOf(hand.handedness);
    ctx.globalAlpha = d.alpha;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.strokeStyle = color;
    ctx.lineWidth = 3 * scale;
    ctx.beginPath();
    for (const [p, q] of CONNECTIONS) { ctx.moveTo(...pts[p]); ctx.lineTo(...pts[q]); }
    ctx.stroke();
    ctx.fillStyle = "#f5f5f7";
    pts.forEach(([x, y], i) => {
      ctx.beginPath();
      ctx.arc(x, y, (i % 4 === 0 ? 4.5 : 3) * scale, 0, Math.PI * 2);
      ctx.fill();
    });
    if (hand.pinch?.active) {
      const [x, y] = [(pts[4][0] + pts[8][0]) / 2, (pts[4][1] + pts[8][1]) / 2];
      ctx.strokeStyle = "#ece8e1";
      ctx.lineWidth = 2 * scale;
      ctx.beginPath();
      ctx.arc(x, y, 14 * scale, 0, Math.PI * 2);
      ctx.stroke();
    }
    // Label plate above the hand: channel dot, name, hold bar.
    const id = hand.pair?.id ?? hand.label;
    const text = labelInfo(id).name;
    const minY = Math.min(...pts.map((p) => p[1]));
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    ctx.font = `600 ${15 * scale}px Inter, sans-serif`;
    const tw = ctx.measureText(text).width + 38 * scale;
    const y = Math.max(minY - 40 * scale, 6);
    ctx.fillStyle = "rgba(16,17,20,0.84)";
    roundRect(cx - tw / 2, y, tw, 30 * scale, 15 * scale);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx - tw / 2 + 15 * scale, y + 15 * scale, 3.5 * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = id === "no_gesture" ? "#a1a1a6" : "#f5f5f7";
    ctx.fillText(text, cx - tw / 2 + 26 * scale, y + 20 * scale);
    if (id !== "no_gesture" && d.holdShown > 0.01) {
      ctx.fillStyle = color;
      roundRect(cx - tw / 2 + 14 * scale, y + 34 * scale, (tw - 28 * scale) * d.holdShown, 3 * scale, 1.5 * scale);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  if (settings.skeleton && latest?.pair && latest.hands.length >= 2) {
    const [p, q] = latest.hands.map((hh) => drawn.get(hh.key)?.pts[0]);
    if (p && q) {
      ctx.setLineDash([6 * scale, 6 * scale]);
      ctx.strokeStyle = "rgba(236,232,225,0.7)";
      ctx.lineWidth = 2 * scale;
      ctx.beginPath();
      ctx.moveTo(...p);
      ctx.lineTo(...q);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
}

function labelInfo(id) {
  if (id?.startsWith("custom:")) {
    const g = custom.items.find((c) => c.id === id);
    return { name: g ? g.name : id, hint: "Your gesture" };
  }
  return gestureInfo(id);
}

// ---------------------------------------------------------------- panels
const perf = { fps: 0, land: 0, cls: 0, history: [] };
const dash = (digits) => (v) => (v < 0.005 ? "–" : v.toFixed(digits));
const metrics = {
  fps: new TweenNumber($("m-fps"), { tau: 0.3, format: dash(0) }),
  fpsTop: new TweenNumber($("fps-top"), { tau: 0.3, format: dash(0) }),
  fpsStage: new TweenNumber($("fps"), { tau: 0.3, format: dash(0) }),
  land: new TweenNumber($("m-land"), { tau: 0.3, format: (v) => (v < 0.05 ? "–" : v >= 100 ? v.toFixed(0) : v.toFixed(1)) }),
  cls: new TweenNumber($("m-cls"), { tau: 0.3, format: (v) => (v < 0.005 ? "–" : v.toFixed(2)) }),
};
const holdRing = new Spring(0, { k: 220, zeta: 1 });
let lastPanel = 0;
let shownNow = "";

// Per-hand prediction cards, keyed by the engine's stable hand key; their bars are spring lists.
const cards = new Map();
function handCard(hand) {
  let card = cards.get(hand.key);
  if (card) return card;
  const el = document.createElement("div");
  el.className = "hand-card";
  el.innerHTML = '<header><b></b><span class="chan"></span></header><div class="probs"></div>';
  const bars = new Map();
  const list = new SpringList(el.querySelector(".probs"), {
    rowHeight: 30,
    max: 5,
    create: (item) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `${icon(item.key, 20, "#d4d4d8")}<div><div class="name">${labelInfo(item.key).name}</div><div class="track"><div class="fill"></div></div></div><span class="pct"></span>`;
      bars.set(item.key, { fill: row.querySelector(".fill"), pct: row.querySelector(".pct"), s: new Spring(0, { k: 160, zeta: 0.9 }) });
      return row;
    },
    update: (row, item, index) => {
      row.classList.toggle("top", index === 0);
      const bar = bars.get(item.key);
      bar.s.set(item.p);
      bar.pct.textContent = `${Math.round(item.p * 100)}%`;
    },
  });
  card = { el, list, bars };
  cards.set(hand.key, card);
  $("hands-view").append(el);
  return card;
}
onFrame((dt) => {
  for (const card of cards.values()) {
    card.list.step(dt);
    for (const [key, bar] of card.bars) {
      if (!card.list.rows.has(key)) { card.bars.delete(key); continue; }
      bar.fill.style.transform = `scaleX(${Math.max(0, bar.s.step(dt)).toFixed(4)})`;
    }
  }
});

function updatePanels(state, t) {
  const primary = state.pair ? null : state.hands[0];
  const id = state.pair?.id ?? primary?.label ?? null;
  const color = state.pair ? "#ece8e1" : channelOf(primary?.handedness);
  els.now.style.setProperty("--hand", color);
  const name = state.hands.length ? labelInfo(id).name : "Show your hand";
  if (name !== shownNow) {
    shownNow = name;
    els.nowName.textContent = name;
    els.nowName.classList.remove("swap");
    void els.nowName.offsetWidth;
    els.nowName.classList.add("swap");
    els.nowIcon.innerHTML = id && id !== "no_gesture" ? icon(id, 32, color) : icon("palm", 32, "rgba(161,161,166,0.3)");
  }
  const conf = state.pair?.confidence ?? primary?.confidence;
  els.nowSub.textContent = state.hands.length
    ? `${state.pair ? "Two hands" : `${primary.handedness} hand`} · ${Math.round((conf ?? 0) * 100)}%`
    : "No hands in view";
  holdRing.set(id === "no_gesture" ? 0 : state.pair?.hold ?? primary?.hold ?? 0);

  metrics.fps.set(perf.fps);
  metrics.fpsTop.set(perf.fps);
  metrics.fpsStage.set(perf.fps);
  metrics.land.set(perf.land);
  metrics.cls.set(perf.cls);

  if (t - lastPanel < 100) return;
  lastPanel = t;
  $("m-hands").textContent = state.hands.length;
  $("hands-count").textContent = `${state.hands.length} hand${state.hands.length === 1 ? "" : "s"}`;
  drawSparkline();

  const live = new Set(state.hands.map((h) => h.pair?.id ?? h.label));
  document.querySelectorAll(".g-card").forEach((c) => c.classList.toggle("live", live.has(c.dataset.g)));
  document.querySelectorAll(".custom-list li[data-g]").forEach((c) => c.classList.toggle("live", live.has(c.dataset.g)));

  $("hands-empty").hidden = state.hands.length > 0;
  const keys = new Set(state.hands.map((h) => h.key));
  for (const [key, card] of cards) if (!keys.has(key)) { card.el.remove(); cards.delete(key); }
  let banner = $("hands-view").querySelector(".pair-banner");
  if (state.pair) {
    if (!banner) {
      banner = document.createElement("div");
      banner.className = "pair-banner";
      $("hands-view").prepend(banner);
    }
    banner.innerHTML = `${icon(state.pair.id, 22)} ${labelInfo(state.pair.id).name} · two hands · ${Math.round(state.pair.confidence * 100)}%`;
  } else banner?.remove();
  for (const h of state.hands) {
    const card = handCard(h);
    card.el.style.setProperty("--c", channelOf(h.handedness));
    card.el.querySelector("b").textContent = labelInfo(h.label).name;
    card.el.querySelector(".chan").textContent = `${h.handedness} hand${h.pinch?.active ? " · pinching" : ""}`;
    // Top guesses at >= 1%, ranked with hysteresis so rows only move when the order genuinely changes.
    let items = stableRank(h.top.filter((c, i) => i === 0 || c.p >= 0.01).map((c) => ({ key: c.id, p: c.p })), [...card.list.rows.keys()]);
    if (h.custom) items = [{ key: h.custom.id, p: h.custom.similarity }, ...items.filter((c) => c.key !== h.custom.id)];
    card.list.set(items);
  }
}
onFrame((dt) => {
  els.hold.style.strokeDasharray = `${(Math.max(0, Math.min(1, holdRing.step(dt))) * 100).toFixed(2)} 100`;
  for (const m of Object.values(metrics)) m.step(dt);
});

function drawSparkline() {
  const c = $("sparkline");
  const dpr = devicePixelRatio || 1;
  const w = (c.width = c.clientWidth * dpr);
  const h = (c.height = 44 * dpr);
  const g = c.getContext("2d");
  const data = perf.history;
  if (data.length < 2) return;
  const max = Math.max(60, ...data);
  const pts = data.map((v, i) => [(i / (data.length - 1)) * w, h - (v / max) * (h - 6 * dpr) - 3 * dpr]);
  const grad = g.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(245,245,247,0.14)");
  grad.addColorStop(1, "rgba(245,245,247,0)");
  g.beginPath();
  pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.lineTo(w, h);
  g.lineTo(0, h);
  g.fillStyle = grad;
  g.fill();
  g.beginPath();
  pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.strokeStyle = "#f5f5f7";
  g.lineWidth = 1.5 * dpr;
  g.stroke();
}

const events = [];
let eventCount = 0;
const logList = new SpringList(els.log, {
  rowHeight: 44,
  max: 10,
  create: (item) => {
    const li = document.createElement("li");
    li.className = "log-row";
    li.innerHTML = `${icon(item.id, 20)}<span>${labelInfo(item.id).name}${item.action ? `<span class="act">→ ${ACTIONS[item.action]}</span>` : ""}</span><time>${item.time}</time>`;
    return li;
  },
});
onFrame((dt) => logList.step(dt));
function logEvent(ev, action) {
  if (ev.id !== "pinch") announce(`${labelInfo(ev.id).name}${action ? `, ${ACTIONS[action]}` : ""}`);
  events.unshift({ key: `${ev.id}@${performance.now()}`, id: ev.id, action, time: new Date().toLocaleTimeString([], { hour12: false }) });
  events.length = Math.min(events.length, 10);
  logList.set(events);
  $("log-empty").hidden = true;
  $("event-count").textContent = String(++eventCount);
}

// ---------------------------------------------------------------- air cursor
let dragTarget = null;
const cur = { x: innerWidth / 2, y: innerHeight / 2, tx: 0, ty: 0, on: false, fresh: true };
let coachShown = store.get(COACH_KEY, false) || params.has("autostart"); // no coach mark in demos / screenshots

function updateCursor(state) {
  const c = state.cursor;
  if (!c) {
    cur.on = false;
    cur.fresh = true;
    dragTarget = null;
    volume.classList.remove("dragging");
    return;
  }
  cur.tx = c.x * innerWidth;
  cur.ty = c.y * innerHeight;
  if (cur.fresh) { cur.x = cur.tx; cur.y = cur.ty; cur.fresh = false; }
  cur.on = true;
  els.cursor.classList.toggle("pinch", c.pinching);
  if (!coachShown && running && teach.phase === "idle") showCoach();
  for (const ev of state.events) {
    if (ev.type === "pinch_start") {
      const el = document.elementFromPoint(cur.x, cur.y);
      const drag = el?.closest("[data-pinch-drag]");
      if (drag) { dragTarget = drag; volume.classList.add("dragging"); }
      else {
        const target = el?.closest("button, [data-action], select, label, a, [role=tab]");
        if (target) {
          // Same press feedback a mouse click gets, so an air click feels like a click.
          target.classList.add("air-press");
          setTimeout(() => target.classList.remove("air-press"), 140);
          target.click();
        }
      }
      logEvent({ id: "pinch" }, null);
      if (!els.coach.hidden) dismissCoach();
    }
    if (ev.type === "pinch_end") { dragTarget = null; volume.classList.remove("dragging"); }
  }
  if (dragTarget && c.pinching) setVolumeFromX(cur.x);
}
onFrame((dt) => {
  // The engine's One Euro filter removes jitter; this upsamples the 30 Hz stream to the display rate.
  const a = approach(dt, tracker.smoothTau * 0.8);
  cur.x += (cur.tx - cur.x) * a;
  cur.y += (cur.ty - cur.y) * a;
  els.cursor.classList.toggle("on", cur.on);
  els.cursor.style.transform = `translate3d(${cur.x.toFixed(1)}px, ${cur.y.toFixed(1)}px, 0)`;
  if (cur.on) {
    const el = document.elementFromPoint(cur.x, cur.y);
    els.cursor.classList.toggle("hover", Boolean(el?.closest("button, [data-action], select, label, a, [role=tab], [data-pinch-drag]")));
  }
  if (!els.coach.hidden) {
    const x = Math.min(innerWidth - 276, Math.max(16, cur.x + 26));
    const y = Math.min(innerHeight - 150, Math.max(16, cur.y + 22));
    els.coach.style.transform = `translate3d(${x}px, ${y}px, 0)`;
  }
});
function showCoach() {
  coachShown = true;
  els.coach.hidden = false;
  $("cursor-toggle").classList.add("nudge");
}
function dismissCoach() {
  els.coach.hidden = true;
  store.set(COACH_KEY, true);
}
$("coach-ok").addEventListener("click", dismissCoach);

// ---------------------------------------------------------------- main loop
let running = false;
let waitingFirst = null; // status label to show once the first tracking result arrives
let lastFrameT = performance.now();

onFrame((dt) => {
  const step = running ? tracker.step() : null;
  if (step) {
    const { state, landmarkMs, t } = step;
    latest = state;
    const frameDt = t - lastFrameT;
    lastFrameT = t;
    perf.fps = perf.fps ? 0.9 * perf.fps + 0.1 * (1000 / Math.max(frameDt, 1)) : 1000 / Math.max(frameDt, 1);
    perf.land = perf.land ? 0.9 * perf.land + 0.1 * landmarkMs : landmarkMs;
    perf.cls = perf.cls ? 0.9 * perf.cls + 0.1 * state.classifyMs : state.classifyMs;
    perf.history.push(perf.fps);
    if (perf.history.length > 120) perf.history.shift();

    for (const ev of state.events) {
      if (ev.type !== "gesture") continue;
      const action = teach.phase === "idle" ? bindings[ev.id] : null; // never fire actions mid-recording
      const ran = action && action !== "none" && runAction(action);
      if (ran) notice(`${labelInfo(ev.id).name} → ${ACTIONS[action]}`, { tone: "live", timeout: 1600 });
      logEvent(ev, ran ? action : null);
      els.now.classList.remove("fired");
      void els.now.offsetWidth;
      els.now.classList.add("fired");
    }

    try {
      lab?.update(
        state.hands.map((h) => h.world),
        state.hands.map((h) => h.landmarks),
        state.hands.map((h) => h.pair?.id ?? h.label),
        state.events,
        settings.mirror,
      );
    } catch (err) {
      console.error("3D lab update failed", err); // never let the 3D view stop recognition
    }
    if (waitingFirst) { setCam(waitingFirst, "live"); waitingFirst = null; }
    trackOverlay(state);
    updatePanels(state, t);
    updateCursor(state);
    updateTeach(state, t);
    window.__gestureStudio = { state, perf: { ...perf, history: undefined } }; // for automated checks
  }
  drawOverlay(dt);
});

// ---------------------------------------------------------------- onboarding
const btnLabel = els.startBtn.querySelector("span:last-child");
const demoBtn = $("demo-btn");
const camStatus = $("status-camera");
let hero = null;

function setCam(text, tone) {
  camStatus.querySelector(".dot").className = `dot ${tone ?? ""}`;
  camStatus.lastChild.textContent = text;
}
function setBoot(text, tone) {
  $("boot-state").textContent = text;
  $("boot-dot").className = `dot ${tone ?? ""}`;
}
function showProblem(problem) {
  $("problem").hidden = !problem;
  if (!problem) return;
  $("problem-title").textContent = problem.title;
  $("problem-body").textContent = problem.body;
}

const ready = tracker.load((text) => setBoot(text, "busy"));
ready
  .then(({ hand, pair, delegate }) => {
    Object.assign(tracker.engine.settings, settings);
    setBoot(`Ready · ${hand.classes.length - 1} + ${pair.classes.length - 1} gestures loaded`, "live");
    const d = $("status-delegate");
    d.textContent = delegate;
    d.classList.toggle("cpu", delegate !== "GPU");
    if (delegate !== "GPU") notice("Running on the CPU: WebGL acceleration isn't available, so tracking is slower.", { timeout: 9000 });
    els.startBtn.disabled = false;
    btnLabel.textContent = "Turn on camera";
    if (params.has("autostart")) begin(); // demos and screenshots
    else session.resumable().then((source) => {
      // Coming from the Studio with the camera on: carry straight on.
      if (!source || running) return;
      setBoot("Resuming", "busy");
      begin(source === "camera" ? undefined : source);
    });
  })
  .catch((err) => {
    console.error(err);
    setBoot("Models failed to load", "err");
    btnLabel.textContent = "Retry";
    els.startBtn.disabled = false;
    els.startBtn.onclick = () => location.reload();
    showProblem({ title: "Couldn't load the recognition models", body: "Check your connection and reload. The models are fetched once, then cached by the browser." });
  });

async function begin(demo = params.get("video") || undefined) {
  if (running) return;
  els.startBtn.disabled = demoBtn.disabled = true;
  btnLabel.textContent = demo ? "Loading clip" : "Waiting for camera";
  setBoot(demo ? "Starting the demo clip" : "Asking for camera permission", "busy");
  setCam(demo ? "Clip" : "Asking", "busy");
  showProblem(null);
  try {
    await ready;
    tracker.stop();
    const source = await tracker.start(demo);
    const { videoWidth: vw, videoHeight: vh } = els.video;
    els.overlay.width = vw;
    els.overlay.height = vh;
    applyCrop();
    els.start.hidden = true;
    hero?.stop();
    els.stage.classList.add("live");
    running = true;
    const isDemo = source === "demo video";
    setCam("Starting tracking", "busy");
    waitingFirst = isDemo ? "Demo" : "Live";
    $("slate-res").textContent = isDemo ? "Demo clip" : `Camera · ${vw} × ${vh}`;
    session.remember(demo ?? "camera");
    $("stop-btn").hidden = false;
    $("stop-btn").lastElementChild.textContent = isDemo ? "Stop demo" : "Stop camera";
    els.nowSub.textContent = "No hands in view";
  } catch (err) {
    console.error(err);
    els.startBtn.disabled = demoBtn.disabled = false;
    btnLabel.textContent = "Try again";
    setBoot("Camera unavailable", "err");
    setCam("Blocked", "err");
    showProblem(describeCameraError(err));
  }
}
els.startBtn.addEventListener("click", () => begin());

/** Turns the camera (or clip) off and returns to the start screen. */
function stopSession() {
  if (teach.phase !== "idle") cancelTeach("Recording stopped with the camera.");
  tracker.stop();
  session.forget();
  running = false;
  waitingFirst = null;
  latest = null;
  drawn.clear();
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  for (const el of [els.video, els.overlay]) el.removeAttribute("style");
  els.stage.style.aspectRatio = "";
  els.stage.classList.remove("live");
  els.start.hidden = false;
  hero?.start();
  cur.on = false;
  $("stop-btn").hidden = true;
  setCam("Standby", "");
  for (const m of Object.values(metrics)) m.set(0);
  updatePanels({ hands: [], pair: null, events: [], classifyMs: 0 }, performance.now() + 1000);
  els.startBtn.disabled = demoBtn.disabled = false;
  btnLabel.textContent = "Turn on camera";
  setBoot("Camera off", "");
  announce("Camera off");
}
$("stop-btn").addEventListener("click", stopSession);
tracker.onEnded = () => {
  running = false;
  setCam("Camera lost", "err");
  notice("The camera stopped. It may have been unplugged or taken by another app.", { tone: "err", timeout: 0, action: { label: "Reconnect", run: () => begin() } });
};
if (params.get("from") === "no-webgl") notice("The 3D studio needs WebGL, which this browser has turned off. Classic works without it.", { timeout: 10000 });
demoBtn.addEventListener("click", () => begin(DEMO_CLIP));

// ---------------------------------------------------------------- keyboard
const toggleHelp = bindHelp($("help"), $("help-btn"));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && teach.phase !== "idle") { cancelTeach(); return; }
  if (typing(e) || e.target.closest?.("[role=slider]")) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  const tabKeys = ["1", "2", "3", "4", "5"];
  if (k === "?") { toggleHelp(); e.preventDefault(); }
  else if (k === "ArrowLeft" && !e.target.closest("[role=tab]")) runAction("prev_slide");
  else if (k === "ArrowRight" && !e.target.closest("[role=tab]")) runAction("next_slide");
  else if (k === "k") runAction("play_pause");
  else if (k === "m") runAction("mute");
  else if (k === "s") runAction("snapshot");
  else if (k === "a") $("cursor-toggle").click();
  else if (k === "x" && running) stopSession();
  else if (tabKeys.includes(k)) selectTab(tabs[Number(k) - 1].dataset.tab);
});

// ---------------------------------------------------------------- boot
settings = { ...settings, ...DEFAULT_SETTINGS, ...store.get(SETTINGS_KEY, {}) };
bindSettings();
syncSettings();
renderDeck();
syncSnaps();
const poses = await loadPoses();
document.querySelectorAll("img[data-glyph]").forEach((img) => (img.src = glyphURL(img.dataset.glyph, 28)));
$("slide-track").replaceChildren();
renderDeck();
deckMotion.slide.jump(deck.slide);
renderGestureGrid();
renderBindings();
renderCustomList();
els.nowIcon.innerHTML = icon("palm", 32, "rgba(161,161,166,0.3)");
if (!els.start.hidden) {
  hero = new PoseHero($("hero"), poses, { name: $("hero-name"), hint: $("hero-hint"), root: $("hero-caption") }, [document.querySelector("#start .start-copy"), $("hero-caption")]);
  hero.start();
}
