// Camera + MediaPipe HandLandmarker + gesture models, shared by the 3D studio and the classic dashboard.
//
// Detection runs in a Web Worker (js/landmark-worker.js) whenever the browser allows it: the page hands it each new
// video frame as an ImageBitmap and keeps rendering at display rate while the model works. Results arrive
// asynchronously and step() hands out the newest one exactly once. If the worker can't start, detection falls back to
// the main thread with the same API.
import { CustomGestures } from "./custom.js";
import { GestureEngine } from "./engine.js";
import { GestureModel } from "./mlp.js";

const BUNDLE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const params = new URLSearchParams(location.search);

export class HandTracker {
  constructor(video) {
    this.video = video;
    this.custom = new CustomGestures();
    this.lastVideoTime = -1;
    this.inFlight = false;
    this.latest = null;
    this.frameId = 0;
    this.interval = 33; // EMA of ms between detection results
    this.lastResultT = 0;
  }

  /**
   * Time constant (s) for easing tracked points between results. It follows the measured detection interval, so the
   * skeleton is still gliding when the next result lands: no stepping at 10 Hz, no lag at 30 Hz.
   */
  get smoothTau() {
    return Math.min(0.12, Math.max(0.035, (this.interval / 1000) * 0.55));
  }

  /** Loads the gesture models and the hand landmarker (worker first; GPU delegate, CPU fallback). */
  async load() {
    const [hand, pair] = await Promise.all([GestureModel.load("models/hand.json"), GestureModel.load("models/pair.json")]);
    this.engine = new GestureEngine(hand, pair, this.custom);
    if (!params.has("mainthread")) {
      try {
        this.delegate = await this.startWorker();
        this.mode = "worker";
      } catch (err) {
        console.warn("Landmark worker unavailable, detecting on the main thread:", err);
        this.worker?.terminate();
        this.worker = null;
      }
    }
    if (!this.worker) {
      this.delegate = await this.loadMainThread();
      this.mode = "main";
    }
    return { hand, pair, delegate: this.delegate };
  }

  startWorker() {
    return new Promise((resolve, reject) => {
      if (typeof Worker === "undefined" || typeof createImageBitmap === "undefined") return reject(new Error("unsupported"));
      const worker = new Worker("js/landmark-worker.js");
      this.worker = worker;
      const timer = setTimeout(() => reject(new Error("worker start timed out")), 45000);
      worker.onerror = (e) => { clearTimeout(timer); reject(new Error(e.message || "worker error")); };
      worker.onmessage = ({ data }) => {
        if (data.type === "ready") { clearTimeout(timer); worker.onmessage = (e) => this.onResult(e.data); resolve(data.delegate); }
        else if (data.type === "error") { clearTimeout(timer); reject(new Error(data.message)); }
      };
      worker.postMessage({ type: "init", prefer: (params.get("delegate") || "GPU").toUpperCase() });
    });
  }

  async loadMainThread() {
    const { FilesetResolver, HandLandmarker } = await import(BUNDLE);
    const vision = await FilesetResolver.forVisionTasks(WASM);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: LANDMARKER_MODEL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
    });
    try {
      this.landmarker = await HandLandmarker.createFromOptions(vision, options("GPU"));
      return "GPU";
    } catch {
      this.landmarker = await HandLandmarker.createFromOptions(vision, options("CPU"));
      return "CPU";
    }
  }

  /**
   * Starts the webcam, or a looping video file: `demo` if given, else `?video=<url>` in the page URL (demos, tests).
   */
  async start(demo = params.get("video")) {
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

  /** Sends the current video frame to the worker if it is new and the worker is idle. */
  pump() {
    if (this.inFlight || this.video.readyState < 2 || this.video.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = this.video.currentTime;
    this.inFlight = true;
    const t = performance.now();
    const id = ++this.frameId;
    this.pending = { id, t };
    // The model works at 224 px; sending 640 px-wide frames instead of a 1280x720 camera frame cuts the copy and
    // transfer cost ~4x without changing the (normalised) landmarks.
    const vw = this.video.videoWidth;
    const scale = vw > 640 ? 640 / vw : 1;
    const size = scale < 1 ? { resizeWidth: 640, resizeHeight: Math.round(this.video.videoHeight * scale), resizeQuality: "medium" } : undefined;
    createImageBitmap(this.video, size)
      .then((bitmap) => this.worker.postMessage({ type: "frame", bitmap, id }, [bitmap]))
      .catch(() => { this.inFlight = false; });
  }

  onResult(data) {
    if (data.type !== "result") return;
    this.inFlight = false;
    const t = this.pending?.id === data.id ? this.pending.t : performance.now();
    this.latest = this.finish(data.hands, t, data.ms);
    this.pump(); // start on the next frame straight away rather than waiting for the next display frame
  }

  /** Runs the (cheap) gesture engine on detected hands. */
  finish(hands, t, landmarkMs) {
    const now = performance.now();
    if (this.lastResultT) this.interval += (Math.min(250, now - this.lastResultT) - this.interval) * 0.15;
    this.lastResultT = now;
    const state = this.engine.update(hands, t, this.aspect);
    state.hands.forEach((h, i) => {
      h.depth = hands[i].depth;
      h.world = hands[i].world;
    });
    return { state, landmarkMs, t };
  }

  /** Returns the newest unconsumed result, or null. Never blocks in worker mode. */
  step() {
    if (this.worker) {
      this.pump();
      const r = this.latest;
      this.latest = null;
      return r;
    }
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
    return this.finish(hands, t0, landmarkMs);
  }
}

/** Plain-language explanation of a getUserMedia failure: { title, body } for the onboarding screen. */
export function describeCameraError(err) {
  switch (err?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        title: "Camera access is blocked",
        body: "Click the camera icon in the address bar, choose Allow, then try again. Or watch the demo to see it working.",
      };
    case "NotFoundError":
    case "OverconstrainedError":
      return { title: "No camera found", body: "Plug in a webcam and try again, or watch the demo instead." };
    case "NotReadableError":
    case "AbortError":
      return { title: "The camera is busy", body: "Another app or tab is using it. Close that, then try again." };
    case "InsecureContextError":
      return { title: "This page can't reach a camera", body: "Open it over https or from localhost. The demo still works." };
    default:
      return { title: "The camera didn't start", body: `${err?.message || "Unknown error"}. Try again, or watch the demo.` };
  }
}
