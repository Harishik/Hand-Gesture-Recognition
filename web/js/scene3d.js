// 3D lab: the tracked hand as a 3D skeleton (MediaPipe world landmarks, in metres) and an object you manipulate
// with gestures: grab (fist) and move to rotate, two open hands apart / together to scale, pinch to change material.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const FINGER_COLORS = [0xf472b6, 0x38bdf8, 0xa78bfa, 0x34d399, 0xfbbf24];
const fingerOf = (i) => (i === 0 ? 0 : Math.floor((i - 1) / 4));
const GRAB = new Set(["fist", "grabbing", "grip"]);
const OPEN = new Set(["palm", "stop", "four", "stop_inverted"]);
const MATERIALS = [
  { color: 0x38bdf8, metalness: 0.6, roughness: 0.25 },
  { color: 0xa78bfa, metalness: 0.2, roughness: 0.5 },
  { color: 0xf472b6, metalness: 0.9, roughness: 0.15 },
  { color: 0x34d399, metalness: 0.4, roughness: 0.35 },
];

export class HandScene {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 10);
    this.camera.position.set(0, 0.05, 0.62);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(0.4, 0.6, 0.8);
    this.scene.add(key);
    const rim = new THREE.PointLight(0xa78bfa, 2.5, 3);
    rim.position.set(-0.5, -0.2, -0.4);
    this.scene.add(rim);

    const grid = new THREE.GridHelper(1.2, 24, 0x38bdf8, 0x1e293b);
    grid.position.y = -0.18;
    grid.material.transparent = true;
    grid.material.opacity = 0.35;
    this.scene.add(grid);

    // Gesture-controlled object.
    this.materialIndex = 0;
    this.object = new THREE.Mesh(
      new THREE.TorusKnotGeometry(0.055, 0.018, 220, 24),
      new THREE.MeshStandardMaterial(MATERIALS[0]),
    );
    this.object.position.set(0.16, 0, 0);
    this.scene.add(this.object);
    this.targetScale = 1;
    this.spin = new THREE.Vector2(0.004, 0.006);

    // Two hand skeletons: 21 joint spheres + 21 bone cylinders each.
    this.hands = [0, 1].map(() => this.makeHand());
    this.grab = null;
    this.twoHandStart = null;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  makeHand() {
    const group = new THREE.Group();
    const joints = Array.from({ length: 21 }, (_, i) => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(i % 4 === 0 ? 0.0065 : 0.005, 16, 12),
        new THREE.MeshStandardMaterial({ color: i === 0 ? 0xffffff : FINGER_COLORS[fingerOf(i)], emissive: i === 0 ? 0x333333 : FINGER_COLORS[fingerOf(i)], emissiveIntensity: 0.6 }),
      );
      group.add(m);
      return m;
    });
    const bones = CONNECTIONS.map(([, b]) => {
      const m = new THREE.Mesh(
        new THREE.CylinderGeometry(0.0028, 0.0028, 1, 10),
        new THREE.MeshStandardMaterial({ color: FINGER_COLORS[fingerOf(b)], emissive: FINGER_COLORS[fingerOf(b)], emissiveIntensity: 0.35, transparent: true, opacity: 0.9 }),
      );
      group.add(m);
      return m;
    });
    group.visible = false;
    this.scene.add(group);
    return { group, joints, bones };
  }

  resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // Narrow (portrait) views: centre the object above the hand instead of beside it.
    const narrow = w / h < 1.2;
    this.object.position.set(narrow ? 0 : 0.16, narrow ? 0.09 : 0, 0);
    this.camera.position.z = narrow ? 0.85 : 0.62;
  }

  /**
   * world: per hand, 21 [x, y, z] in metres (hand-centred, from MediaPipe). screen: per hand, 21 [x, y] display
   * coordinates in [0, 1]. labels: per hand, recognised gesture id. mirror: whether the 2D view is mirrored.
   */
  update(world, screen, labels, events, mirror) {
    const up = new THREE.Vector3(0, 1, 0);
    this.hands.forEach((hand, h) => {
      const pts = world[h];
      hand.group.visible = Boolean(pts);
      if (!pts) return;
      // Place each hand where it is on screen, so two hands appear side by side.
      const ox = (screen[h][0][0] - 0.5) * 0.5;
      const oy = -(screen[h][0][1] - 0.5) * 0.35;
      const k = 1.8; // enlarge the real-size hand (~18 cm) for the scene
      const v = pts.map(([x, y, z]) => new THREE.Vector3((mirror ? -x : x) * k + ox, -y * k + oy, -z * k));
      v.forEach((p, i) => hand.joints[i].position.copy(p));
      CONNECTIONS.forEach(([a, b], i) => {
        const bone = hand.bones[i];
        const dir = new THREE.Vector3().subVectors(v[b], v[a]);
        bone.position.copy(v[a]).addScaledVector(dir, 0.5);
        bone.scale.set(1, dir.length(), 1);
        bone.quaternion.setFromUnitVectors(up, dir.normalize());
      });
    });

    // Grab with a fist: the object rotates with the hand's movement on screen.
    const grabbing = labels.findIndex((l) => GRAB.has(l));
    if (grabbing >= 0) {
      const p = screen[grabbing][9];
      if (this.grab) {
        this.object.rotation.y += (p[0] - this.grab[0]) * 7;
        this.object.rotation.x += (p[1] - this.grab[1]) * 7;
        this.spin.set(0, 0);
      }
      this.grab = [p[0], p[1]];
      this.object.material.emissive?.setHex(0x0b3b52);
    } else {
      this.grab = null;
      this.object.material.emissive?.setHex(0x000000);
      this.spin.set(0.004, 0.006);
    }

    // Two open hands: the change in distance between them scales the object.
    if (screen.length >= 2 && OPEN.has(labels[0]) && OPEN.has(labels[1])) {
      const d = Math.hypot(screen[0][9][0] - screen[1][9][0], screen[0][9][1] - screen[1][9][1]);
      if (!this.twoHandStart) this.twoHandStart = { d, scale: this.targetScale };
      this.targetScale = Math.min(2.4, Math.max(0.4, this.twoHandStart.scale * (d / Math.max(this.twoHandStart.d, 1e-3))));
    } else {
      this.twoHandStart = null;
    }

    if (events.some((e) => e.type === "pinch_start")) {
      this.materialIndex = (this.materialIndex + 1) % MATERIALS.length;
      const { color, metalness, roughness } = MATERIALS[this.materialIndex];
      this.object.material.color.setHex(color);
      this.object.material.metalness = metalness;
      this.object.material.roughness = roughness;
    }
  }

  render() {
    this.object.rotation.x += this.spin.x;
    this.object.rotation.y += this.spin.y;
    const s = this.object.scale.x + (this.targetScale - this.object.scale.x) * 0.15;
    this.object.scale.setScalar(s);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
