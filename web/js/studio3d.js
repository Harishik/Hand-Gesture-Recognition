// Gesture Studio: the live view is a Three.js scene. The camera feed sits on a screen in 3D space with viewfinder
// marks; tracked hands are drawn over it as 3D skeletons with real depth, coloured by channel (left / right);
// a ring of pose glyphs shows every gesture; a matte object responds to grab, two-hand scale and pinch.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { drawGlyph, glyphCanvas, loadPoses } from "./glyph.js";
import { GESTURES, gestureInfo } from "./gestures-meta.js";
import { HandTracker } from "./tracker.js";

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const BONE = new THREE.Color(0xece8e1);
const CHANNEL = { Left: "#7fa8ff", Right: "#ffb25c" };
const channelOf = (handedness) => CHANNEL[handedness] ?? "#ece8e1";
const GRAB = new Set(["fist", "grabbing", "grip"]);
const OPEN = new Set(["palm", "stop", "four", "stop_inverted"]);
const SCREEN_H = 2.1;
const SCREEN_Y = 0.25;
const nameOf = (id) => gestureInfo(id).name;

const $ = (id) => document.getElementById(id);
const viewport = $("viewport");
const canvas = $("scene");

// ------------------------------------------------------------------ renderer and post-processing
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.25)); // leave GPU headroom for hand tracking
renderer.toneMapping = THREE.ACESFilmicToneMapping;
const scene = new THREE.Scene();
// Studio cyclorama: a soft vertical gradient (painted in-scene; the bloom pipeline renders an opaque frame).
{
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 512;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 512);
  grad.addColorStop(0, "#1e2126");
  grad.addColorStop(0.45, "#30353d");
  grad.addColorStop(0.7, "#2b2f36");
  grad.addColorStop(1, "#1a1d21");
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 512);
  const bg = new THREE.CanvasTexture(c);
  bg.colorSpace = THREE.SRGBColorSpace;
  scene.background = bg;
}
const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 50);
const cameraBase = new THREE.Vector3(0, 0.1, 4.6);
camera.position.copy(cameraBase);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.32, 0.4, 0.82); // only bright hand strokes glow
composer.addPass(bloom);
composer.addPass(new OutputPass());

scene.add(new THREE.HemisphereLight(0xf3efe8, 0x23272d, 1.1));
const key = new THREE.DirectionalLight(0xfff1e0, 2.2);
key.position.set(3, 4, 5);
scene.add(key);
const fill = new THREE.DirectionalLight(0x9fb8ff, 0.8);
fill.position.set(-4, 1, 2);
scene.add(fill);

// Soft contact shadow under the stage.
{
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, "rgba(0,0,0,0.45)");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(9, 2.2), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.set(0, -1.62, -0.6);
  scene.add(shadow);
}

// ------------------------------------------------------------------ camera screen with viewfinder marks
const video = $("video");
const videoTexture = new THREE.VideoTexture(video);
videoTexture.colorSpace = THREE.SRGBColorSpace;
const screen = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: videoTexture, color: 0xc9c6c0, toneMapped: false }));
screen.position.set(0, SCREEN_Y, 0);
screen.visible = false;
scene.add(screen);

const marks = new THREE.Group();
scene.add(marks);
function buildMarks(w, h) {
  marks.clear();
  const L = 0.16;
  const pts = [];
  for (const [sx, sy] of [[-1, 1], [1, 1], [-1, -1], [1, -1]]) {
    const x = (sx * w) / 2 + sx * 0.05;
    const y = SCREEN_Y + (sy * h) / 2 + sy * 0.05;
    pts.push(x, y, 0, x - sx * L, y, 0, x, y, 0, x, y - sy * L, 0);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  marks.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xece8e1, transparent: true, opacity: 0.55 })));
}

let screenW = SCREEN_H * (16 / 9);
function layoutScreen(aspect, mirror) {
  screenW = SCREEN_H * aspect;
  screen.scale.set(screenW, SCREEN_H, 1);
  videoTexture.wrapS = THREE.RepeatWrapping;
  videoTexture.repeat.x = mirror ? -1 : 1;
  videoTexture.offset.x = mirror ? 1 : 0;
  buildMarks(screenW, SCREEN_H);
  fit();
}

/** Display landmark (x, y in [0, 1], MediaPipe relative depth z) -> point on / in front of the screen. */
function toScene(x, y, z, out) {
  return out.set((x - 0.5) * screenW, -(y - 0.5) * SCREEN_H + SCREEN_Y, 0.03 - z * screenW * 0.9);
}

// ------------------------------------------------------------------ hands
const jointGeo = new THREE.SphereGeometry(1, 14, 10);
const boneGeo = new THREE.CylinderGeometry(1, 1, 1, 8, 1, true);
const up = new THREE.Vector3(0, 1, 0);
const tmpM = new THREE.Matrix4();
const tmpV = new THREE.Vector3();
const TRAIL = 24;

function makeHand() {
  const group = new THREE.Group();
  const boneMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
  const joints = new THREE.InstancedMesh(jointGeo, new THREE.MeshBasicMaterial({ color: BONE, toneMapped: false }), 21);
  group.add(joints);
  const bones = CONNECTIONS.map(() => {
    const m = new THREE.Mesh(boneGeo, boneMat);
    group.add(m);
    return m;
  });
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, toneMapped: false }));
  trail.frustumCulled = false;
  scene.add(trail, group);
  const label = makeLabel();
  scene.add(label.sprite);
  group.visible = trail.visible = false;
  return { group, joints, bones, boneMat, trail, trailPts: [], label, pts: Array.from({ length: 21 }, () => new THREE.Vector3()) };
}
const hands3d = [makeHand(), makeHand()];

function updateHand(h3, hand, color) {
  const k = screenW / 3.7;
  hand.landmarks.forEach(([x, y], i) => toScene(x, y, hand.depth?.[i] ?? 0, h3.pts[i]));
  h3.pts.forEach((p, i) => {
    const r = ([4, 8, 12, 16, 20].includes(i) ? 0.03 : i === 0 ? 0.034 : 0.022) * k;
    h3.joints.setMatrixAt(i, tmpM.makeScale(r, r, r).setPosition(p));
  });
  h3.joints.instanceMatrix.needsUpdate = true;
  h3.boneMat.color.set(color);
  CONNECTIONS.forEach(([a, b], i) => {
    const bone = h3.bones[i];
    const dir = tmpV.subVectors(h3.pts[b], h3.pts[a]);
    const len = dir.length();
    bone.position.copy(h3.pts[a]).addScaledVector(dir, 0.5);
    bone.scale.set(0.0095 * k, len, 0.0095 * k);
    bone.quaternion.setFromUnitVectors(up, dir.normalize());
  });
  h3.trailPts.push(h3.pts[8].clone());
  if (h3.trailPts.length > TRAIL) h3.trailPts.shift();
  const arr = h3.trail.geometry.attributes.position.array;
  for (let i = 0; i < TRAIL; i++) {
    const p = h3.trailPts[Math.max(0, h3.trailPts.length - TRAIL + i)] ?? h3.pts[8];
    arr.set([p.x, p.y, p.z], i * 3);
  }
  h3.trail.geometry.attributes.position.needsUpdate = true;
  h3.trail.material.color.set(color);
  h3.group.visible = h3.trail.visible = true;

  const top = h3.pts.reduce((a, p) => (p.y > a.y ? p : a));
  const cx = h3.pts.reduce((a, p) => a + p.x, 0) / 21;
  h3.label.sprite.position.set(cx, top.y + 0.3, top.z + 0.2);
}

function hideHand(h3) {
  h3.group.visible = h3.trail.visible = h3.label.sprite.visible = false;
  h3.trailPts = [];
}

// ------------------------------------------------------------------ in-scene labels
function makeLabel() {
  const c = document.createElement("canvas");
  c.width = 560;
  c.height = 132;
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, toneMapped: false }));
  sprite.scale.set(1.5, 0.354, 1);
  sprite.renderOrder = 10;
  return { g: c.getContext("2d"), texture, sprite, key: "" };
}

function drawLabel(label, name, color, conf, hold, active) {
  const key = `${name}|${color}|${Math.round(conf * 100)}|${Math.round(hold * 24)}|${active}`;
  label.sprite.visible = true;
  if (key === label.key) return;
  label.key = key;
  const g = label.g;
  g.clearRect(0, 0, 560, 132);
  g.fillStyle = "rgba(30,33,38,0.92)";
  g.beginPath();
  g.roundRect(4, 4, 552, 96, 18);
  g.fill();
  g.fillStyle = color;
  g.beginPath();
  g.roundRect(4, 4, 8, 96, [18, 0, 0, 18]);
  g.fill();
  g.fillStyle = active ? "#ece8e1" : "#8e939c";
  g.font = "700 46px 'Bricolage Grotesque', sans-serif";
  g.fillText(name, 34, 66);
  g.fillStyle = "#8e939c";
  g.font = "500 30px 'IBM Plex Sans', sans-serif";
  g.textAlign = "right";
  g.fillText(`${Math.round(conf * 100)}%`, 532, 64);
  g.textAlign = "left";
  if (hold > 0 && active) {
    g.fillStyle = color;
    g.fillRect(34, 112, 492 * hold, 6);
  }
  label.texture.needsUpdate = true;
}

// ------------------------------------------------------------------ pose-glyph ring
const RING_IDS = Object.keys(GESTURES).filter((id) => !GESTURES[id].motion && id !== "no_gesture");
const ring = new THREE.Group();
ring.position.set(0, -1.5, -1.3);
scene.add(ring);
const RADIUS = 2.7;
let tiles = [];

function paintTile(tile, color, active) {
  const { g, id } = tile.userData;
  g.clearRect(0, 0, 256, 300);
  g.fillStyle = active ? "#ece8e1" : "#2b2f36";
  g.beginPath();
  g.roundRect(4, 4, 248, 292, 22);
  g.fill();
  drawGlyph(g, id, 34, 22, 188, { color: active ? "#1e2126" : "#ece8e1", width: 5 });
  g.fillStyle = active ? "#1e2126" : "#8e939c";
  g.font = "600 26px 'IBM Plex Sans', sans-serif";
  g.textAlign = "center";
  const name = nameOf(id);
  g.fillText(name.length > 15 ? `${name.slice(0, 14)}…` : name, 128, 268);
  if (active) {
    g.fillStyle = color;
    g.fillRect(96, 280, 64, 6);
  }
  tile.material.map.needsUpdate = true;
}

function buildRing() {
  tiles = RING_IDS.map((id, i) => {
    const c = document.createElement("canvas");
    c.width = 256;
    c.height = 300;
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.4, 0.47), new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false }));
    const angle = (i / RING_IDS.length) * Math.PI * 2;
    mesh.position.set(Math.sin(angle) * RADIUS, 0, Math.cos(angle) * RADIUS);
    mesh.rotation.y = angle;
    mesh.userData = { id, angle, g: c.getContext("2d"), active: false, lift: 0 };
    paintTile(mesh, null, false);
    ring.add(mesh);
    return mesh;
  });
}
let ringTarget = 0;
let ringIdle = true;
function focusTile(id) {
  const tile = tiles.find((t) => t.userData.id === id);
  if (!tile) return;
  const want = -tile.userData.angle;
  ringTarget = want + Math.round((ring.rotation.y - want) / (Math.PI * 2)) * Math.PI * 2;
  ringIdle = false;
}

// ------------------------------------------------------------------ the object (grab / scale / pinch)
const FINISHES = [
  { color: 0xece8e1, roughness: 0.55, metalness: 0.05 },
  { color: 0xffb25c, roughness: 0.35, metalness: 0.2 },
  { color: 0x7fa8ff, roughness: 0.3, metalness: 0.35 },
  { color: 0x3a3f47, roughness: 0.25, metalness: 0.8 },
];
const object = new THREE.Group();
object.position.set(2.85, SCREEN_Y - 0.1, 0.1);
scene.add(object);
const core = new THREE.Mesh(new THREE.TorusKnotGeometry(0.32, 0.1, 220, 28), new THREE.MeshStandardMaterial(FINISHES[0]));
object.add(core);
const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.48, 0.06, 64), new THREE.MeshStandardMaterial({ color: 0x2b2f36, roughness: 0.8 }));
pedestal.position.set(2.85, -0.78, 0.1);
scene.add(pedestal);
const objectState = { grab: null, twoStart: null, scale: 1, target: 1, finish: 0, spin: 0.004 };

function updateObject(state) {
  const labels = state.hands.map((h) => h.label);
  const gi = labels.findIndex((l) => GRAB.has(l));
  if (gi >= 0) {
    const p = state.hands[gi].landmarks[9];
    if (objectState.grab) {
      object.rotation.y += (p[0] - objectState.grab[0]) * 9;
      object.rotation.x += (p[1] - objectState.grab[1]) * 9;
    }
    objectState.grab = [p[0], p[1]];
    objectState.spin = 0;
  } else {
    objectState.grab = null;
    objectState.spin = 0.004;
  }
  if (state.hands.length >= 2 && OPEN.has(labels[0]) && OPEN.has(labels[1])) {
    const [a, b] = state.hands;
    const d = Math.hypot(a.landmarks[9][0] - b.landmarks[9][0], a.landmarks[9][1] - b.landmarks[9][1]);
    objectState.twoStart ??= { d, scale: objectState.target };
    objectState.target = THREE.MathUtils.clamp(objectState.twoStart.scale * (d / Math.max(objectState.twoStart.d, 1e-3)), 0.45, 1.8);
  } else {
    objectState.twoStart = null;
  }
  if (state.events.some((e) => e.type === "pinch_start")) {
    objectState.finish = (objectState.finish + 1) % FINISHES.length;
    const f = FINISHES[objectState.finish];
    core.material.color.setHex(f.color);
    core.material.roughness = f.roughness;
    core.material.metalness = f.metalness;
  }
}

// ------------------------------------------------------------------ confirmation pulse (expanding ring at the palm)
const pulses = [];
function pulse(at, color) {
  const m = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.112, 64), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, toneMapped: false, side: THREE.DoubleSide }));
  m.position.copy(at);
  scene.add(m);
  pulses.push({ m, life: 1 });
}
function updatePulses(dt) {
  for (let i = pulses.length - 1; i >= 0; i--) {
    const p = pulses[i];
    p.life -= dt * 2.2;
    const s = 1 + (1 - p.life) * 3.5;
    p.m.scale.set(s, s, s);
    p.m.material.opacity = Math.max(p.life, 0) * 0.9;
    if (p.life <= 0) {
      scene.remove(p.m);
      p.m.geometry.dispose();
      p.m.material.dispose();
      pulses.splice(i, 1);
    }
  }
}

// ------------------------------------------------------------------ sizing
function fit() {
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  camera.aspect = w / h;
  const wide = w / h > 1.25;
  object.visible = pedestal.visible = wide;
  const needW = screenW + (wide ? 3.9 : 0.5);
  const needH = SCREEN_H + 2.4;
  const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  cameraBase.x = wide ? 0.95 : 0;
  cameraBase.z = Math.max(needH / 2 / tan, needW / 2 / (tan * camera.aspect));
  camera.updateProjectionMatrix();
}
new ResizeObserver(fit).observe(viewport);
layoutScreen(16 / 9, true);

// ------------------------------------------------------------------ panel
const hud = { fps: 0, land: 0, last: 0 };
const nowGlyph = $("now-glyph").getContext("2d");
let shownId = null;

function setNow(id, color) {
  if (id === shownId && color === setNow.color) return;
  shownId = id;
  setNow.color = color;
  nowGlyph.clearRect(0, 0, 176, 176);
  if (id && id !== "no_gesture") drawGlyph(nowGlyph, id, 16, 16, 144, { color, width: 6 });
}

function updatePanel(state, t, landmarkMs) {
  hud.land = hud.land ? 0.9 * hud.land + 0.1 * (landmarkMs + state.classifyMs) : landmarkMs + state.classifyMs;
  if (t - hud.last < 100) return;
  hud.last = t;
  $("fps").textContent = hud.fps.toFixed(0);
  $("latency").textContent = hud.land.toFixed(0);

  const primary = state.hands[0];
  const id = state.pair?.id ?? primary?.label ?? null;
  const color = state.pair ? "#ece8e1" : channelOf(primary?.handedness);
  document.querySelector(".now").style.setProperty("--hand", color);
  setNow(id, color);
  if (!primary) {
    $("now-name").textContent = "No hand";
    $("now-detail").textContent = "Raise a hand in front of the camera";
    $("now-conf").textContent = "0";
    $("now-meter").style.width = "0";
    $("now-hold").style.width = "0";
    $("alts").innerHTML = "";
    return;
  }
  const conf = state.pair?.confidence ?? primary.confidence;
  $("now-name").textContent = id === "no_gesture" ? "No gesture" : nameOf(id);
  $("now-detail").textContent = state.pair
    ? "Two-hand gesture"
    : `${primary.handedness} hand${state.hands.length > 1 ? ", plus one more" : ""}`;
  $("now-conf").textContent = Math.round(conf * 100);
  $("now-meter").style.width = `${conf * 100}%`;
  $("now-hold").style.width = `${(state.pair?.hold ?? primary.hold ?? 0) * 100}%`;

  const alts = primary.top.filter((c) => c.id !== id).slice(0, 4);
  $("alts").replaceChildren(
    ...alts.map((c) => {
      const li = document.createElement("li");
      li.append(glyphCanvas(c.id, 28, { color: "#8e939c", width: 1.4, joints: false }));
      li.insertAdjacentHTML("beforeend", `<span>${c.id === "no_gesture" ? "No gesture" : nameOf(c.id)}</span><span class="pct">${Math.round(c.p * 100)}%</span><span class="bar-track"><i style="width:${(c.p * 100).toFixed(1)}%"></i></span>`);
      return li;
    }),
  );
}

function logGesture(id, color) {
  const list = $("history");
  list.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  li.append(glyphCanvas(id, 28, { color, width: 1.4, joints: false }));
  li.insertAdjacentHTML("beforeend", `<span>${nameOf(id)}</span><time>${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time>`);
  list.prepend(li);
  while (list.children.length > 6) list.lastChild.remove();
  const now = document.querySelector(".now");
  now.classList.remove("fired");
  void now.offsetWidth;
  now.classList.add("fired");
}

// ------------------------------------------------------------------ loop
const tracker = new HandTracker(video);
const clock = new THREE.Clock();
const camTarget = new THREE.Vector3();
let latest = null;
let lastFrame = performance.now();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const step = tracker.engine ? tracker.step() : null;

  if (step) {
    const { state, landmarkMs, t } = step;
    latest = state;
    const frameDt = t - lastFrame;
    lastFrame = t;
    hud.fps = hud.fps ? 0.9 * hud.fps + 0.1 * (1000 / Math.max(frameDt, 1)) : 1000 / Math.max(frameDt, 1);

    hands3d.forEach((h3, i) => {
      const hand = state.hands[i];
      if (!hand) return hideHand(h3);
      const color = channelOf(hand.handedness);
      updateHand(h3, hand, color);
      const id = hand.pair?.id ?? hand.label;
      const active = id !== "no_gesture";
      drawLabel(h3.label, active ? nameOf(id) : "No gesture", color, hand.pair?.confidence ?? hand.confidence, hand.pair?.hold ?? hand.hold ?? 0, active);
    });

    const active = state.pair?.id ?? state.hands[0]?.label;
    if (active && active !== "no_gesture") focusTile(active);
    else ringIdle = true;
    for (const tile of tiles) {
      const on = tile.userData.id === active;
      if (on !== tile.userData.active) {
        tile.userData.active = on;
        paintTile(tile, state.pair ? "#ece8e1" : channelOf(state.hands[0]?.handedness), on);
      }
    }

    updateObject(state);
    for (const ev of state.events) {
      if (ev.type !== "gesture") continue;
      const idx = Math.max(0, state.hands.findIndex((h) => h.handedness === ev.source));
      const color = ev.source === "two-hand" ? "#ece8e1" : channelOf(state.hands[idx]?.handedness);
      if (state.hands[idx]) pulse(hands3d[idx].pts[9].clone(), color);
      if (ev.id === "swipe_left") ringTarget -= 0.7;
      if (ev.id === "swipe_right") ringTarget += 0.7;
      if (!ev.id.startsWith("swipe")) logGesture(ev.id, color);
    }
    updatePanel(state, t, landmarkMs);
  }

  if (ringIdle) ringTarget += dt * 0.08;
  ring.rotation.y += (ringTarget - ring.rotation.y) * Math.min(1, dt * 4);
  for (const tile of tiles) {
    tile.userData.lift += ((tile.userData.active ? 1 : 0) - tile.userData.lift) * Math.min(1, dt * 8);
    const s = 1 + tile.userData.lift * 0.45;
    tile.scale.set(s, s, s);
    tile.position.y = tile.userData.lift * 0.2;
  }

  object.rotation.y += objectState.spin;
  object.rotation.x += objectState.spin * 0.4;
  objectState.scale += (objectState.target - objectState.scale) * 0.15;
  object.scale.setScalar(objectState.scale);
  updatePulses(dt);

  const c = latest?.cursor;
  camTarget.set(cameraBase.x + (c ? (c.x - 0.5) * 0.35 : 0), cameraBase.y + (c ? (0.5 - c.y) * 0.2 : 0), cameraBase.z);
  camera.position.lerp(camTarget, 0.04);
  camera.lookAt(cameraBase.x * 0.85, SCREEN_Y - 0.35, 0);
  composer.render();
}

// ------------------------------------------------------------------ boot
const btn = $("start-btn");
Promise.all([tracker.load(), loadPoses(), document.fonts.ready])
  .then(([{ delegate }]) => {
    buildRing();
    btn.disabled = false;
    btn.textContent = "Turn on camera";
    if (new URLSearchParams(location.search).has("autostart")) queueMicrotask(() => btn.click()); // demos and screenshots
    $("start-hint").textContent = `Hand tracking runs on your ${delegate === "GPU" ? "graphics card" : "processor"}. Works best with your hand about an arm's length away.`;
  })
  .catch((err) => {
    btn.textContent = "Couldn't load the models";
    $("start-hint").textContent = "Check your connection and reload the page.";
    console.error(err);
  });
animate();

btn.addEventListener("click", async () => {
  btn.disabled = true;
  btn.textContent = "Starting camera";
  try {
    const source = await tracker.start();
    layoutScreen(tracker.aspect, tracker.engine.settings.mirror);
    screen.visible = true;
    $("start").hidden = true;
    $("cam-dot").classList.add("live");
    $("cam-state").textContent = source === "demo video" ? "Demo video" : "Camera on";
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "Turn on camera";
    if (new URLSearchParams(location.search).has("autostart")) queueMicrotask(() => btn.click()); // demos and screenshots
    $("start-hint").textContent = "Camera access was blocked. Allow it in the address bar, then try again.";
    console.error(err);
  }
});

window.__gestureStudio3d = () => ({ fps: hud.fps, latencyMs: hud.land, hands: latest?.hands.map((h) => h.label) ?? [] });
