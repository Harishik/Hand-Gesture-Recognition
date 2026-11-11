// Gesture Studio: the live view is a Three.js scene. The camera feed sits on a screen in 3D space with viewfinder
// marks; tracked hands are drawn over it as 3D skeletons with real depth, coloured by channel (left / right);
// a ring of pose glyphs shows every gesture; a matte object responds to grab, two-hand scale and pinch.
//
// Motion model: the tracker delivers ~30 Hz. Everything it touches only sets *targets*; the render loop eases every
// visible quantity towards its target per display frame with frame-rate independent springs / exponentials
// (js/motion.js), so nothing on screen steps at camera rate.
import * as THREE from "three";
import { drawGlyph, glyphCanvas, glyphURL, loadPoses } from "./glyph.js";
import { GESTURES, gestureInfo } from "./gestures-meta.js";
import { PoseHero } from "./hero.js";
import { approach, onFrame, reducedMotion, Spring, SpringList, stableRank, TweenNumber } from "./motion.js";
import { describeCameraError, HandTracker } from "./tracker.js";
import { bindHelp, notice, typing } from "./ui.js";

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
const DEMO_CLIP = "test-media/demo.webm";
const nameOf = (id) => (id === "no_gesture" ? "No gesture" : gestureInfo(id).name);
const params = new URLSearchParams(location.search);

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
  grad.addColorStop(0, "#09090b");
  grad.addColorStop(0.45, "#16171b");
  grad.addColorStop(0.7, "#121316");
  grad.addColorStop(1, "#08080a");
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 512);
  const bg = new THREE.CanvasTexture(c);
  bg.colorSpace = THREE.SRGBColorSpace;
  scene.background = bg;
}
const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 50);
const cameraBase = new THREE.Vector3(0, 0.1, 4.6);
camera.position.copy(cameraBase);

// No post-processing: a bloom pass cost ~4x the frame time and fought hand tracking for the GPU (measured p95
// 133 ms vs 17 ms without). Glow is painted instead: additive halo sprites behind lit tiles.
const HALO = (() => {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, "rgba(255,255,255,0.55)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.16)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
})();

scene.add(new THREE.HemisphereLight(0xf3efe8, 0x15181c, 1.0));
const key = new THREE.DirectionalLight(0xfff1e0, 2.2);
key.position.set(3, 4, 5);
scene.add(key);
const fill = new THREE.DirectionalLight(0x9fb8ff, 0.8);
fill.position.set(-4, 1, 2);
scene.add(fill);

// Floor: a faint capture-volume grid that fades into the dark, plus a soft contact shadow under the stage.
{
  const grid = new THREE.GridHelper(14, 28, 0x1f2126, 0x16171a);
  grid.position.set(0, -1.62, -0.6);
  grid.material.transparent = true;
  grid.material.opacity = 0.45;
  grid.material.depthWrite = false;
  scene.add(grid);
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, "rgba(0,0,0,0.5)");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(9, 2.2), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.set(0, -1.61, -0.6);
  scene.add(shadow);
}

// ------------------------------------------------------------------ camera screen with viewfinder marks
const video = $("video");
const videoTexture = new THREE.VideoTexture(video);
videoTexture.colorSpace = THREE.SRGBColorSpace;
const screenMat = new THREE.MeshBasicMaterial({ map: videoTexture, color: 0xc9c6c0, toneMapped: false, transparent: true, opacity: 0 });
const screen = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), screenMat);
screen.position.set(0, SCREEN_Y, 0);
screen.visible = false;
scene.add(screen);
const screenIn = new Spring(0, { k: 60, zeta: 1 }); // the feed fades and settles in when the camera starts

const marks = new THREE.Group();
scene.add(marks);
const marksMat = new THREE.LineBasicMaterial({ color: 0xece8e1, transparent: true, opacity: 0.55 });
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
  marks.add(new THREE.LineSegments(g, marksMat));
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
  const boneMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false, transparent: true, opacity: 0 });
  const jointMat = new THREE.MeshBasicMaterial({ color: BONE, toneMapped: false, transparent: true, opacity: 0 });
  const joints = new THREE.InstancedMesh(jointGeo, jointMat, 21);
  group.add(joints);
  const bones = CONNECTIONS.map(() => {
    const m = new THREE.Mesh(boneGeo, boneMat);
    group.add(m);
    return m;
  });
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, toneMapped: false }));
  trail.frustumCulled = false;
  scene.add(trail, group);
  const label = makeLabel();
  scene.add(label.sprite);
  group.visible = trail.visible = false;
  return { group, joints, jointMat, bones, boneMat, trail, trailPts: [], label, alpha: 0, present: false,
    color: new THREE.Color(0xece8e1), targetColor: new THREE.Color(0xece8e1),
    pts: Array.from({ length: 21 }, () => new THREE.Vector3()),
    target: Array.from({ length: 21 }, () => new THREE.Vector3()), fresh: true };
}
const hands3d = [makeHand(), makeHand()];

// Tracking arrives at camera rate (~30 Hz) but we render at display rate: updateHand only sets targets, and drawHand
// glides the skeleton towards them every frame, so motion stays fluid instead of stepping.
function updateHand(h3, hand, color) {
  hand.landmarks.forEach(([x, y], i) => toScene(x, y, hand.depth?.[i] ?? 0, h3.target[i]));
  if (h3.fresh) {
    h3.pts.forEach((p, i) => p.copy(h3.target[i])); // first sighting: no fly-in from the origin
    h3.color.set(color);
  }
  h3.fresh = false;
  h3.present = true;
  h3.targetColor.set(color);
  h3.group.visible = h3.trail.visible = true;
}

function drawHand(h3, dt) {
  if (!h3.group.visible) return;
  // Presence fades in over ~80 ms and out over ~160 ms, so a hand never pops in or vanishes.
  h3.alpha += ((h3.present ? 1 : 0) - h3.alpha) * approach(dt, h3.present ? 0.08 : 0.16);
  if (!h3.present && h3.alpha < 0.01) {
    h3.fresh = true;
    h3.group.visible = h3.trail.visible = h3.label.sprite.visible = false;
    h3.trailPts = [];
    return;
  }
  const a = approach(dt, tracker.smoothTau); // adapts to the detection rate: continuous motion between results
  h3.pts.forEach((p, i) => p.lerp(h3.target[i], a));
  h3.color.lerp(h3.targetColor, approach(dt, 0.12));
  const k = screenW / 3.7;
  h3.pts.forEach((p, i) => {
    const r = ([4, 8, 12, 16, 20].includes(i) ? 0.03 : i === 0 ? 0.034 : 0.022) * k;
    h3.joints.setMatrixAt(i, tmpM.makeScale(r, r, r).setPosition(p));
  });
  h3.joints.instanceMatrix.needsUpdate = true;
  h3.boneMat.color.copy(h3.color);
  h3.boneMat.opacity = h3.jointMat.opacity = h3.alpha;
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
  h3.trail.material.color.copy(h3.color);
  h3.trail.material.opacity = 0.55 * h3.alpha;

  const top = h3.pts.reduce((a, p) => (p.y > a.y ? p : a));
  const cx = h3.pts.reduce((a, p) => a + p.x, 0) / 21;
  h3.label.sprite.position.set(cx, top.y + 0.3, top.z + 0.2);
  h3.label.sprite.material.opacity = h3.alpha;
  // The hold bar on the label fills smoothly between tracker frames.
  h3.label.hold += (h3.label.holdTarget - h3.label.hold) * approach(dt, 0.06);
  paintLabel(h3.label);
}

function hideHand(h3) {
  h3.present = false; // drawHand fades it out, then hides it
}

// ------------------------------------------------------------------ in-scene labels
function makeLabel() {
  const c = document.createElement("canvas");
  c.width = 560;
  c.height = 132;
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, toneMapped: false, opacity: 0 }));
  sprite.scale.set(1.5, 0.354, 1);
  sprite.renderOrder = 10;
  return { g: c.getContext("2d"), texture, sprite, key: "", hold: 0, holdTarget: 0 };
}

function setLabel(label, name, color, conf, hold, active) {
  Object.assign(label, { name, color, conf, active, holdTarget: active ? hold : 0 });
  if (!active) label.hold = 0;
  label.sprite.visible = true;
}

function paintLabel(label) {
  const { name, color, conf, active } = label;
  const hold = label.hold;
  const key = `${name}|${color}|${Math.round(conf * 100)}|${Math.round(hold * 96)}|${active}`;
  if (key === label.key) return;
  label.key = key;
  const g = label.g;
  g.clearRect(0, 0, 560, 132);
  g.fillStyle = "rgba(16,17,20,0.86)";
  g.beginPath();
  g.roundRect(4, 4, 552, 96, 48);
  g.fill();
  g.strokeStyle = "rgba(255,255,255,0.08)";
  g.lineWidth = 2;
  g.stroke();
  g.fillStyle = color;
  g.beginPath();
  g.arc(44, 52, 9, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = active ? "#f5f5f7" : "#a1a1a6";
  g.font = "600 44px Inter, sans-serif";
  g.letterSpacing = "-1.5px";
  g.fillText(name, 70, 67);
  g.letterSpacing = "0px";
  g.fillStyle = "#8e8e93";
  g.font = "500 30px Inter, sans-serif";
  g.textAlign = "right";
  g.fillText(`${Math.round(conf * 100)}%`, 516, 64);
  g.textAlign = "left";
  if (hold > 0.005 && active) {
    g.fillStyle = "rgba(255,255,255,0.1)";
    g.beginPath(); g.roundRect(48, 114, 464, 5, 3); g.fill();
    g.fillStyle = color;
    g.beginPath(); g.roundRect(48, 114, Math.max(5, 464 * hold), 5, 3); g.fill();
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

// Each tile is two planes: the idle face, and the lit face in front of it. The lit face's opacity follows the tile's
// lift spring, so activation crossfades instead of repainting in one frame.
function paintFace(g, id, color, active) {
  g.clearRect(0, 0, 256, 300);
  g.fillStyle = active ? "#f5f5f7" : "rgba(28,29,33,0.92)";
  g.beginPath();
  g.roundRect(4, 4, 248, 292, 34);
  g.fill();
  if (!active) {
    g.strokeStyle = "rgba(255,255,255,0.07)";
    g.lineWidth = 2;
    g.stroke();
  }
  drawGlyph(g, id, 44, 30, 168, { color: active ? "#0b0b0d" : "#d4d4d8", width: 4.5 });
  g.fillStyle = active ? "#0b0b0d" : "#8e8e93";
  g.font = "500 25px Inter, sans-serif";
  g.textAlign = "center";
  const name = nameOf(id);
  g.fillText(name.length > 15 ? `${name.slice(0, 14)}…` : name, 128, 268);
  if (active && color) {
    g.fillStyle = color;
    g.beginPath(); g.roundRect(108, 278, 40, 6, 3); g.fill();
  }
}

function faceMesh(id, color, active) {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 300;
  const g = c.getContext("2d");
  paintFace(g, id, color, active);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.4, 0.47), new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false, depthWrite: !active }));
  mesh.userData.g = g;
  return mesh;
}

function buildRing() {
  tiles = RING_IDS.map((id, i) => {
    const tile = new THREE.Group();
    const idle = faceMesh(id, null, false);
    const lit = faceMesh(id, "#ece8e1", true);
    lit.position.z = 0.002;
    lit.material.opacity = 0;
    const halo = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.2), new THREE.MeshBasicMaterial({ map: HALO, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
    halo.position.z = -0.01;
    tile.add(halo, idle, lit);
    const angle = (i / RING_IDS.length) * Math.PI * 2;
    tile.position.set(Math.sin(angle) * RADIUS, 0, Math.cos(angle) * RADIUS);
    tile.rotation.y = angle;
    tile.userData = { id, angle, lit, halo, active: false, color: "#ece8e1", lift: new Spring(0, { k: 150, zeta: 0.62 }) };
    ring.add(tile);
    return tile;
  });
}
function setTileActive(tile, on, color) {
  const u = tile.userData;
  if (on && color !== u.color) {
    paintFace(u.lit.userData.g, u.id, color, true);
    u.lit.material.map.needsUpdate = true;
    u.color = color;
  }
  u.active = on;
  u.lift.set(on ? 1 : 0);
}

// Ring physics: a critically-damped spring with a speed cap, so it glides and settles instead of snapping.
let ringTarget = 0;
let ringVel = 0;
let ringIdle = true;
let stableId = null;
let stableSince = 0;
const RING_K = 9; // spring stiffness (1/s^2)
const RING_C = 2 * Math.sqrt(RING_K) * 0.95; // just under critical damping: one soft overshoot
const RING_VMAX = 1.6; // rad/s
function stepRing(dt) {
  const acc = RING_K * (ringTarget - ring.rotation.y) - RING_C * ringVel;
  ringVel = Math.max(-RING_VMAX, Math.min(RING_VMAX, ringVel + acc * dt));
  ring.rotation.y += ringVel * dt;
}
function snapRing() {
  const step = (Math.PI * 2) / Math.max(1, tiles.length);
  ringTarget = Math.round(ringTarget / step) * step;
}
function focusTile(id) {
  const tile = tiles.find((t) => t.userData.id === id);
  if (!tile) return;
  const want = -tile.userData.angle;
  ringTarget = want + Math.round((ring.rotation.y - want) / (Math.PI * 2)) * Math.PI * 2;
  ringIdle = false;
}
function spinRing(dir, t = performance.now()) {
  ringVel += dir * 1.2; // momentum from the flick
  ringTarget += dir * ((Math.PI * 2) / Math.max(1, tiles.length)) * 3;
  snapRing();
  ringIdle = false;
  stableSince = t + 1500; // let the swipe play out before a held gesture re-centres
}

// ------------------------------------------------------------------ the object (grab / scale / pinch)
const FINISHES = [
  { name: "Bone matte", color: 0xece8e1, roughness: 0.55, metalness: 0.05 },
  { name: "Amber satin", color: 0xffb25c, roughness: 0.35, metalness: 0.2 },
  { name: "Cobalt gloss", color: 0x7fa8ff, roughness: 0.3, metalness: 0.35 },
  { name: "Graphite metal", color: 0x3a3f47, roughness: 0.25, metalness: 0.8 },
];
const object = new THREE.Group();
object.position.set(2.85, SCREEN_Y - 0.1, 0.1);
scene.add(object);
const core = new THREE.Mesh(new THREE.TorusKnotGeometry(0.32, 0.1, 220, 28), new THREE.MeshStandardMaterial(FINISHES[0]));
object.add(core);
const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.48, 0.06, 64), new THREE.MeshStandardMaterial({ color: 0x1d2228, roughness: 0.8 }));
pedestal.position.set(2.85, -0.78, 0.1);
scene.add(pedestal);
// Tracker frames only move these targets; the render loop eases the object towards them every frame.
const objectState = {
  grab: null, twoStart: null, finish: 0,
  rotX: 0, rotY: 0, // target rotation (rad)
  spin: new Spring(1, { k: 20, zeta: 1 }), // idle auto-rotation, ramps instead of switching on/off
  scale: new Spring(1, { k: 90, zeta: 0.7 }),
  finishColor: new THREE.Color(FINISHES[0].color), finishRough: FINISHES[0].roughness, finishMetal: FINISHES[0].metalness,
  bump: new Spring(0, { k: 260, zeta: 0.45 }), // a small squash when the finish changes
};

function setFinish(i) {
  objectState.finish = (i + FINISHES.length) % FINISHES.length;
  const f = FINISHES[objectState.finish];
  objectState.finishColor.setHex(f.color);
  objectState.finishRough = f.roughness;
  objectState.finishMetal = f.metalness;
  objectState.bump.velocity += 4;
}

function updateObject(state) {
  const labels = state.hands.map((h) => h.label);
  const gi = labels.findIndex((l) => GRAB.has(l));
  if (gi >= 0) {
    const p = state.hands[gi].landmarks[9];
    if (objectState.grab) {
      objectState.rotY += (p[0] - objectState.grab[0]) * 9;
      objectState.rotX += (p[1] - objectState.grab[1]) * 9;
    }
    objectState.grab = [p[0], p[1]];
    objectState.spin.set(0);
  } else {
    objectState.grab = null;
    objectState.spin.set(1);
  }
  if (state.hands.length >= 2 && OPEN.has(labels[0]) && OPEN.has(labels[1])) {
    const [a, b] = state.hands;
    const d = Math.hypot(a.landmarks[9][0] - b.landmarks[9][0], a.landmarks[9][1] - b.landmarks[9][1]);
    objectState.twoStart ??= { d, scale: objectState.scale.target };
    objectState.scale.set(THREE.MathUtils.clamp(objectState.twoStart.scale * (d / Math.max(objectState.twoStart.d, 1e-3)), 0.45, 1.8));
  } else {
    objectState.twoStart = null;
  }
  if (state.events.some((e) => e.type === "pinch_start")) setFinish(objectState.finish + 1);
}

function stepObject(dt) {
  // Idle spin: 0.24 rad/s around y (≈ the old 0.004 rad per 60 Hz frame), scaled by the spin spring.
  const spin = Math.max(0, objectState.spin.step(dt)) * (reducedMotion.matches ? 0 : 1);
  objectState.rotY += 0.24 * spin * dt;
  objectState.rotX += 0.096 * spin * dt;
  const a = approach(dt, 0.07);
  object.rotation.y += (objectState.rotY - object.rotation.y) * a;
  object.rotation.x += (objectState.rotX - object.rotation.x) * a;
  const s = objectState.scale.step(dt) * (1 + 0.06 * objectState.bump.step(dt));
  object.scale.setScalar(Math.max(0.05, s));
  const f = approach(dt, 0.18);
  core.material.color.lerp(objectState.finishColor, f);
  core.material.roughness += (objectState.finishRough - core.material.roughness) * f;
  core.material.metalness += (objectState.finishMetal - core.material.metalness) * f;
}

// ------------------------------------------------------------------ confirmation pulse (expanding ring at the palm)
const pulses = [];
function pulse(at, color) {
  if (reducedMotion.matches) return;
  const m = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.112, 64), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, toneMapped: false, side: THREE.DoubleSide }));
  m.position.copy(at);
  scene.add(m);
  pulses.push({ m, life: 1 });
}
function updatePulses(dt) {
  for (let i = pulses.length - 1; i >= 0; i--) {
    const p = pulses[i];
    p.life -= dt * 2.2;
    const e = 1 - Math.max(p.life, 0) ** 3; // ease-out expansion
    const s = 1 + e * 3.5;
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
// The inspector floats over the right of the stage and the bar over its top, so the camera's view is offset to
// compose the scene into the space that is actually free.
function freeArea() {
  const panel = $("inspector");
  const floating = getComputedStyle(panel).position === "absolute";
  return { right: floating ? panel.offsetWidth + 24 : 0, top: 56 }; // the bar always floats over the top
}
function fit() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);

  const free = freeArea();
  camera.setViewOffset(w, h, free.right / 2, -free.top / 2, w, h);
  const fw = w - free.right;
  const fh = h - free.top;
  camera.aspect = w / h;
  const wide = fw / fh > 1.25;
  object.visible = pedestal.visible = wide;
  const needW = screenW + (wide ? 3.9 : 0.5);
  const needH = SCREEN_H + 2.4;
  const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  cameraBase.x = wide ? 0.95 : 0;
  cameraBase.z = Math.max(needH / 2 / tan / (fh / h), needW / 2 / (tan * camera.aspect * (fw / w)));
  camera.updateProjectionMatrix();
}
new ResizeObserver(fit).observe(canvas);
layoutScreen(16 / 9, true);

// ------------------------------------------------------------------ inspector
const hud = { fps: 0, land: 0, last: 0, frames: 0 };
const nowGlyph = $("now-glyph").getContext("2d");
const nowEl = $("now");
const dash = (v) => (v < 0.5 ? "–" : Math.round(v).toString());
const ui = {
  conf: new TweenNumber($("now-conf"), { tau: 0.1 }),
  fps: new TweenNumber($("fps"), { tau: 0.4, format: dash }),
  latency: new TweenNumber($("latency"), { tau: 0.4, format: dash }),
  confArc: new Spring(0, { k: 160, zeta: 1 }),
  holdArc: new Spring(0, { k: 220, zeta: 1 }),
  altBars: new Map(),
};
let shownId;

function setNow(id, color) {
  if (id === shownId && color === setNow.color) return;
  const changed = id !== shownId;
  shownId = id;
  setNow.color = color;
  nowGlyph.clearRect(0, 0, 176, 176);
  if (id && id !== "no_gesture") drawGlyph(nowGlyph, id, 8, 8, 160, { color, width: 6 });
  else drawGlyph(nowGlyph, "palm", 16, 16, 144, { color: "rgba(161,161,166,0.28)", width: 4.5, joints: false }); // idle silhouette, never an empty box
  if (changed) {
    const h2 = $("now-name");
    h2.classList.remove("swap");
    void h2.offsetWidth;
    h2.classList.add("swap");
  }
}

const altList = new SpringList($("alts"), {
  rowHeight: 40,
  gap: 2,
  max: 4,
  create: (item) => {
    const li = document.createElement("li");
    li.className = "alt-row";
    li.append(glyphCanvas(item.key, 24, { color: "#d4d4d8", width: 1.3, joints: false }));
    li.insertAdjacentHTML("beforeend", `<span>${nameOf(item.key)}</span><span class="pct"></span><span class="bar2"><i></i></span>`);
    ui.altBars.set(item.key, { el: li.querySelector(".bar2 i"), pct: li.querySelector(".pct"), s: new Spring(0, { k: 140, zeta: 0.9 }) });
    return li;
  },
  update: (li, item) => {
    const bar = ui.altBars.get(item.key);
    bar.s.set(item.p);
    bar.pct.textContent = `${Math.round(item.p * 100)}%`;
  },
});

const takeList = new SpringList($("history"), {
  rowHeight: 36,
  gap: 0,
  max: 6,
  create: (item) => {
    const li = document.createElement("li");
    li.className = "take-row";
    li.append(glyphCanvas(item.id, 24, { color: item.color, width: 1.3, joints: false }));
    li.insertAdjacentHTML("beforeend", `<span><i class="chip" style="background:${item.color}"></i><b>${nameOf(item.id)}</b></span><time>${item.time}</time>`);
    return li;
  },
});
const takes = [];

function renderTips() {
  // Three gestures to try, different each time the inspector goes idle.
  const pool = ["peace", "ok", "like", "rock", "call", "palm", "one", "three_gun", "fist", "thumb_index"];
  const pick = pool.sort(() => Math.random() - 0.5).slice(0, 3);
  $("tips").innerHTML = pick
    .map((id, i) => `<li style="--i:${i}"><img src="${glyphURL(id, 32, "#d4d4d8")}" alt=""><b>${nameOf(id)}</b><span>${gestureInfo(id).hint}</span></li>`)
    .join("");
}

let idle = null;
let posesReady = false;
function setIdle(on) {
  if (on === idle) return;
  idle = on;
  $("alts-idle").hidden = !on;
  $("alts").hidden = on;
  if (on) $("alts-note").hidden = true;
  if (on) {
    altList.clear();
    if (posesReady) renderTips();
  }
}

function updatePanel(state, t, landmarkMs) {
  const ms = landmarkMs + state.classifyMs;
  hud.frames += 1;
  // Skip the first frames: they include model and GPU warm-up and would poison the running average for seconds.
  if (hud.frames > 10) hud.land = hud.land ? 0.9 * hud.land + 0.1 * ms : ms;
  ui.fps.set(hud.fps);
  if (hud.land) ui.latency.set(hud.land);

  const primary = state.hands[0];
  const id = state.pair?.id ?? primary?.label ?? null;
  const color = state.pair ? "#ece8e1" : channelOf(primary?.handedness);
  nowEl.style.setProperty("--hand", color);
  setNow(id, color);
  const chan = $("now-chan");
  if (!primary) {
    $("now-name").textContent = "No hand";
    $("now-detail").textContent = "Raise a hand in front of the camera";
    chan.textContent = "No hand";
    chan.className = "chan";
    ui.conf.set(0);
    ui.confArc.set(0);
    ui.holdArc.set(0);
    setIdle(true);
    return;
  }
  setIdle(false);
  const conf = state.pair?.confidence ?? primary.confidence;
  // The name and channel must agree with the 3D label on every frame, so these are never throttled.
  $("now-name").textContent = nameOf(id);
  chan.textContent = state.pair ? "Both hands" : primary.handedness === "Left" ? "Left hand" : primary.handedness === "Right" ? "Right hand" : "Hand";
  chan.className = `chan ${state.pair ? "both" : (primary.handedness || "").toLowerCase()}`;
  ui.conf.set(conf * 100);
  ui.confArc.set(conf);
  ui.holdArc.set(id === "no_gesture" ? 0 : state.pair?.hold ?? primary.hold ?? 0);

  // Secondary text and the alternatives list update at 10 Hz; their motion is interpolated every frame anyway.
  if (t - hud.last < 100) return;
  hud.last = t;
  $("now-detail").textContent = state.pair
    ? "Two-hand gesture"
    : `${gestureInfo(id).hint ?? ""}${state.hands.length > 1 ? " · second hand in view" : ""}`;
  // Only real contenders (>= 1%) are listed; a wall of 0% rows is noise, not information.
  const alts = stableRank(primary.top.filter((c) => c.id !== id && c.p >= 0.01).map((c) => ({ key: c.id, p: c.p })), [...altList.rows.keys()]).slice(0, 4);
  altList.set(alts);
  $("alts-note").hidden = alts.length > 0;
}

function logGesture(id, color) {
  const time = new Date().toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  takes.unshift({ key: `${id}@${performance.now()}`, id, color, time });
  takes.length = Math.min(takes.length, 6);
  $("history-empty").hidden = true;
  takeList.set(takes);
  takeCount += 1;
  $("take-count").textContent = String(takeCount);
  nowEl.classList.remove("fired");
  void nowEl.offsetWidth;
  nowEl.classList.add("fired");
}
let takeCount = 0;

onFrame((dt) => {
  ui.conf.step(dt);
  ui.fps.step(dt);
  ui.latency.step(dt);
  const c = Math.max(0, Math.min(1, ui.confArc.step(dt)));
  const h = Math.max(0, Math.min(1, ui.holdArc.step(dt)));
  $("conf-bar").style.transform = `scaleX(${c.toFixed(4)})`;
  $("hold-bar").style.transform = `scaleX(${h.toFixed(4)})`;
  for (const bar of ui.altBars.values()) bar.el.style.transform = `scaleX(${Math.max(0, bar.s.step(dt)).toFixed(4)})`;
  altList.step(dt);
  takeList.step(dt);
  for (const key of ui.altBars.keys()) if (!altList.rows.has(key)) ui.altBars.delete(key);
});

// ------------------------------------------------------------------ slate: resolution and running timecode
let liveSince = 0;
function updateTimecode(now) {
  if (!liveSince) return;
  const s = (now - liveSince) / 1000;
  const p = (n) => String(Math.floor(n)).padStart(2, "0");
  $("timecode").textContent = `${p(s / 3600)}:${p((s / 60) % 60)}:${p(s % 60)}:${p((s % 1) * 30)}`;
}

// ------------------------------------------------------------------ loop
const tracker = new HandTracker(video);
const clock = new THREE.Clock();
const camTarget = new THREE.Vector3();
let latest = null;
let lastFrame = performance.now();
let running = false;
let waitingFirst = null;
// While the opaque start screen covers the stage there is nothing to see: skip rendering, and use the idle frames to
// upload tile textures a few at a time so the reveal doesn't hitch on ~80 uploads in one frame.
let stageVisible = false;
const warmQueue = [];

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const step = running ? tracker.step() : null;

  if (step) {
    const { state, landmarkMs, t } = step;
    latest = state;
    if (waitingFirst) { $("cam-state").textContent = waitingFirst; $("cam-dot").className = "dot live"; waitingFirst = null; }
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
      setLabel(h3.label, nameOf(id), color, hand.pair?.confidence ?? hand.confidence, hand.pair?.hold ?? hand.hold ?? 0, active);
    });

    const active = state.pair?.id ?? state.hands[0]?.label;
    // Only turn the ring once a gesture has been steady for 300 ms; single-frame flickers are ignored.
    if (active !== stableId) { stableId = active; stableSince = t; }
    if (active && active !== "no_gesture") { if (t - stableSince > 300) focusTile(active); }
    else if (t - stableSince > 1200) ringIdle = true;
    const tileColor = state.pair ? "#ece8e1" : channelOf(state.hands[0]?.handedness);
    for (const tile of tiles) {
      const on = tile.userData.id === active;
      if (on !== tile.userData.active || (on && tileColor !== tile.userData.color)) setTileActive(tile, on, tileColor);
    }

    updateObject(state);
    for (const ev of state.events) {
      if (ev.type !== "gesture") continue;
      const idx = Math.max(0, state.hands.findIndex((h) => h.handedness === ev.source));
      const color = ev.source === "two-hand" ? "#ece8e1" : channelOf(state.hands[idx]?.handedness);
      if (state.hands[idx]) pulse(hands3d[idx].pts[9].clone(), color);
      if (ev.id === "swipe_left" || ev.id === "swipe_right") spinRing(ev.id === "swipe_left" ? -1 : 1, t);
      if (!ev.id.startsWith("swipe")) logGesture(ev.id, color);
    }
    updatePanel(state, t, landmarkMs);
  }

  if (ringIdle && !reducedMotion.matches) ringTarget += dt * 0.035; // slow showroom drift
  stepRing(Math.min(dt, 0.05));
  for (const tile of tiles) {
    const lift = tile.userData.lift.step(dt);
    const s = 1 + lift * 0.45;
    tile.scale.set(s, s, s);
    tile.position.y = lift * 0.2;
    tile.userData.lit.material.opacity = Math.max(0, Math.min(1, lift));
    tile.userData.halo.material.opacity = Math.max(0, Math.min(1, lift)) * 0.55;
  }

  stepObject(dt);
  updatePulses(dt);

  // Camera feed settles in when it starts.
  const sIn = screenIn.step(dt);
  screenMat.opacity = Math.max(0, Math.min(1, sIn));
  marksMat.opacity = 0.55 * Math.max(0, Math.min(1, sIn));
  screen.position.z = (1 - sIn) * -0.25;

  // Parallax follows the air cursor with a ~0.45 s lag, and drifts back to centre when the hand leaves.
  const c = latest?.cursor;
  camTarget.set(cameraBase.x + (c ? (c.x - 0.5) * 0.35 : 0), cameraBase.y + (c ? (0.5 - c.y) * 0.2 : 0), cameraBase.z);
  camera.position.lerp(camTarget, reducedMotion.matches ? 1 : approach(dt, 0.45));
  camera.lookAt(cameraBase.x * 0.85, SCREEN_Y - 0.35, 0);
  hands3d.forEach((h3) => drawHand(h3, dt));
  if (!stageVisible) {
    for (let i = 0; i < 3 && warmQueue.length; i++) renderer.initTexture(warmQueue.shift());
    return;
  }
  renderer.render(scene, camera);
  updateTimecode(performance.now());
}

// ------------------------------------------------------------------ onboarding and boot
const btn = $("start-btn");
const btnLabel = btn.querySelector("span:last-child");
const demoBtn = $("demo-btn");
const startEl = $("start");
let hero = null;

function showProblem(problem) {
  const box = $("problem");
  box.hidden = !problem;
  if (!problem) return;
  $("problem-title").textContent = problem.title;
  $("problem-body").textContent = problem.body;
}

function setBoot(text, tone) {
  $("boot-state").textContent = text;
  $("boot-dot").className = `dot ${tone ?? ""}`;
}

setIdle(true);

const ready = Promise.all([tracker.load(), loadPoses(), document.fonts.ready]);
loadPoses().then((poses) => {
  posesReady = true;
  renderTips(); // glyphs need the pose data
  setNow(null, "#ece8e1");
  hero = new PoseHero($("hero"), poses, { name: $("hero-name"), hint: $("hero-hint"), root: $("hero-caption") }, [document.querySelector("#start .start-copy"), $("hero-caption")]);
  hero.start();
  document.querySelectorAll(".controls .g").forEach((el) => {
    el.innerHTML = `<img src="${glyphURL(el.dataset.glyph, 24, "#d4d4d8")}" alt="" width="24" height="24">`;
  });
});

ready
  .then(([{ delegate }]) => {
    buildRing();
    ring.traverse((o) => o.material?.map && warmQueue.push(o.material.map));
    if (params.has("autostart")) stageVisible = true;
    canvas.classList.add("ready");
    btn.disabled = false;
    btnLabel.textContent = "Turn on camera";
    setBoot("Ready · models loaded", "live");
    const dev = $("device");
    dev.textContent = delegate === "GPU" ? "GPU" : "CPU";
    dev.classList.toggle("cpu", delegate !== "GPU");
    $("start-hint").textContent = delegate === "GPU"
      ? "Hand tracking runs on your graphics card. Keep your hand about an arm's length from the camera."
      : "No usable GPU, so hand tracking runs on the processor. It still works, at a lower frame rate.";
    if (delegate !== "GPU") notice("Running on the CPU: WebGL acceleration isn't available in this browser.", { tone: "busy", timeout: 9000 });
    if (params.has("autostart")) queueMicrotask(() => begin()); // demos and screenshots
    else btn.focus({ preventScroll: true });
  })
  .catch((err) => {
    console.error(err);
    btnLabel.textContent = "Retry";
    btn.disabled = false;
    btn.onclick = () => location.reload();
    setBoot("Models failed to load", "err");
    showProblem({ title: "Couldn't load the recognition models", body: "Check your connection and reload. The models (about 8 MB) are fetched once, then cached by the browser." });
  });
animate();

async function begin(demo = params.get("video") || undefined) {
  btn.disabled = demoBtn.disabled = true;
  btnLabel.textContent = demo ? "Loading clip" : "Waiting for camera";
  setBoot(demo ? "Starting the demo clip" : "Asking for camera permission", "busy");
  showProblem(null);
  try {
    await ready;
    tracker.stop();
    const source = await tracker.start(demo);
    layoutScreen(tracker.aspect, tracker.engine.settings.mirror);
    screen.visible = true;
    screenIn.set(1);
    running = true;
    liveSince = performance.now();
    stageVisible = true;
    renderer.compile(scene, camera);
    startEl.classList.add("leaving");
    viewport.classList.add("live");
    setTimeout(() => { startEl.hidden = true; hero?.stop(); }, 600);
    const isDemo = source === "demo video";
    $("cam-state").textContent = "Starting tracking";
    $("cam-dot").className = "dot busy";
    waitingFirst = isDemo ? "Demo" : "Live";
    $("slate-res").textContent = `${video.videoWidth} × ${video.videoHeight}${isDemo ? " · demo clip" : ""}`;
    fit();
    $("inspector").focus({ preventScroll: true });
  } catch (err) {
    console.error(err);
    btn.disabled = demoBtn.disabled = false;
    btnLabel.textContent = "Try again";
    setBoot("Camera unavailable", "err");
    $("cam-dot").className = "dot err";
    showProblem(describeCameraError(err));
  }
}

btn.addEventListener("click", () => begin());
demoBtn.addEventListener("click", () => begin(DEMO_CLIP));

// ------------------------------------------------------------------ keyboard
const toggleHelp = bindHelp($("help"), $("help-btn"));
document.addEventListener("keydown", (e) => {
  if (typing(e)) return;
  const k = e.key;
  if (k === "?") { toggleHelp(); e.preventDefault(); }
  else if (k === "ArrowLeft" || k === "ArrowRight") { spinRing(k === "ArrowLeft" ? -1 : 1); e.preventDefault(); }
  else if (k === "f" || k === "F") setFinish(objectState.finish + 1), notice(`Finish: ${FINISHES[objectState.finish].name}`, { tone: "live", timeout: 1600 });
  else if (k === "r" || k === "R") { objectState.rotX = objectState.rotY = 0; objectState.scale.set(1); }
  else if (k === "c" || k === "C") location.href = "classic.html";
  else if (k === "Enter" && !startEl.hidden && !btn.disabled && document.activeElement === document.body) begin();
});

window.__gestureStudio3d = () => ({ fps: hud.fps, latencyMs: hud.land, hands: latest?.hands.map((h) => h.label) ?? [] });
