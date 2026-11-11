// Camera + MediaPipe HandLandmarker + gesture models, shared by the 3D studio and the classic dashboard.
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { CustomGestures } from "./custom.js";
import { GestureEngine } from "./engine.js";
import { GestureModel } from "./mlp.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export class HandTracker {
  constructor(video) {
    this.video = video;
    this.custom = new CustomGestures();
    this.lastVideoTime = -1;
  }

  /** Loads the gesture models and the hand landmarker (GPU delegate, CPU fallback). */
  async load() {
    const [hand, pair, vision] = await Promise.all([
      GestureModel.load("models/hand.json"),
      GestureModel.load("models/pair.json"),
      FilesetResolver.forVisionTasks(WASM),
    ]);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: LANDMARKER_MODEL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
    });
    try {
      this.landmarker = await HandLandmarker.createFromOptions(vision, options("GPU"));
      this.delegate = "GPU";
    } catch {
      this.landmarker = await HandLandmarker.createFromOptions(vision, options("CPU"));
      this.delegate = "CPU";
    }
    this.engine = new GestureEngine(hand, pair, this.custom);
    return { hand, pair, delegate: this.delegate };
  }

  /**
   * Starts the webcam, or a looping video file: `demo` if given, else `?video=<url>` in the page URL (demos, tests).
   */
  async start(demo = new URLSearchParams(location.search).get("video")) {
    if (demo) {
      this.video.srcObject = null;
      this.video.src = demo;
      this.video.loop = true;
    } else {
      if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error("No camera API"), { name: "InsecureContextError" });
      this.video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
    }
    this.video.muted = true;
    this.video.playsInline = true;
    await this.video.play();
    if (this.video.readyState < 2) await new Promise((r) => this.video.addEventListener("loadeddata", r, { once: true }));
    this.source = demo ? "demo video" : `${this.video.videoWidth}×${this.video.videoHeight}`;
    return this.source;
  }

  /** Stops the camera stream, if one is running. */
  stop() {
    this.video.srcObject?.getTracks?.().forEach((t) => t.stop());
  }

  get aspect() {
    return this.video.videoWidth / this.video.videoHeight || 16 / 9;
  }

  /** Processes the current video frame if it is new. Returns null when there is no new frame. */
  step() {
    if (this.video.currentTime === this.lastVideoTime || this.video.readyState < 2) return null;
    this.lastVideoTime = this.video.currentTime;
    const t0 = performance.now();
    const result = this.landmarker.detectForVideo(this.video, t0);
    const landmarkMs = performance.now() - t0;
    const handedness = result.handedness ?? result.handednesses ?? [];
    const hands = result.landmarks.map((lm, i) => ({
      landmarks: lm.map((p) => [p.x, p.y]),
      depth: lm.map((p) => p.z),
      world: result.worldLandmarks?.[i]?.map((p) => [p.x, p.y, p.z]),
      handedness: handedness[i]?.[0]?.categoryName ?? "Hand",
    }));
    const state = this.engine.update(hands, t0, this.aspect);
    state.hands.forEach((h, i) => {
      h.depth = hands[i].depth;
      h.world = hands[i].world;
    });
    return { state, landmarkMs, t: t0 };
  }
}

/** Plain-language explanation of a getUserMedia failure: { title, body, retry } for the onboarding screen. */
export function describeCameraError(err) {
  switch (err?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        title: "Camera access is blocked",
        body: "Click the camera icon in the address bar, choose Allow, then try again. Or play the demo clip to see it working.",
      };
    case "NotFoundError":
    case "OverconstrainedError":
      return { title: "No camera found", body: "Plug in a webcam and try again, or play the demo clip instead." };
    case "NotReadableError":
    case "AbortError":
      return { title: "The camera is busy", body: "Another app or tab is using it. Close that, then try again." };
    case "InsecureContextError":
      return { title: "This page can't reach a camera", body: "Open it over https or from localhost. The demo clip still works." };
    default:
      return { title: "The camera didn't start", body: `${err?.message || "Unknown error"}. Try again, or play the demo clip.` };
  }
}
