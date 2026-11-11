// Hand landmark detection off the main thread. The page posts video frames as ImageBitmaps; this worker runs
// MediaPipe HandLandmarker (GPU delegate via OffscreenCanvas, CPU fallback) and posts landmarks back, so rendering,
// input and animation on the main thread never wait for the model.
// A classic (not module) worker: MediaPipe's WASM loader relies on importScripts, which module workers forbid.
// The library itself is ES-module only, so it is loaded with dynamic import().
const BUNDLE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
let landmarker = null;
let lastTs = 0;

async function init(prefer = "GPU") {
  const stage = (text) => self.postMessage({ type: "stage", text });
  stage("Loading the hand tracker");
  const { FilesetResolver, HandLandmarker } = await import(BUNDLE);
  const vision = await FilesetResolver.forVisionTasks(WASM);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
  });
  const order = prefer === "CPU" ? ["CPU", "GPU"] : ["GPU", "CPU"];
  let delegate;
  stage("Loading the hand model");
  try {
    landmarker = await HandLandmarker.createFromOptions(vision, options(order[0]));
    delegate = order[0];
  } catch {
    landmarker = await HandLandmarker.createFromOptions(vision, options(order[1]));
    delegate = order[1];
  }
  stage(delegate === "GPU" ? "Preparing the graphics card" : "Preparing the processor");
  // Warm-up: the first inference compiles GPU shaders and can take many seconds. Do it now, while the page still
  // says "Loading", instead of after the camera starts.
  try {
    const canvas = new OffscreenCanvas(256, 256);
    const g = canvas.getContext("2d");
    g.fillStyle = "#808080";
    g.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 2; i++) {
      lastTs += 1;
      landmarker.detectForVideo(canvas.transferToImageBitmap ? canvas.transferToImageBitmap() : canvas, lastTs);
      g.fillRect(0, 0, 256, 256);
    }
  } catch (err) {
    console.warn("warm-up skipped", err);
  }
  return delegate;
}

self.onmessage = async ({ data }) => {
  if (data.type === "init") {
    try {
      self.postMessage({ type: "ready", delegate: await init(data.prefer) });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err?.message || err) });
    }
    return;
  }
  if (data.type === "frame") {
    const { bitmap, id } = data;
    const ts = Math.max(performance.now(), lastTs + 1); // VIDEO mode needs strictly increasing timestamps
    lastTs = ts;
    const t0 = performance.now();
    let result;
    try {
      result = landmarker.detectForVideo(bitmap, ts);
    } finally {
      bitmap.close();
    }
    const handedness = result.handedness ?? result.handednesses ?? [];
    self.postMessage({
      type: "result",
      id,
      ms: performance.now() - t0,
      hands: result.landmarks.map((lm, i) => ({
        landmarks: lm.map((p) => [p.x, p.y]),
        depth: lm.map((p) => p.z),
        world: result.worldLandmarks?.[i]?.map((p) => [p.x, p.y, p.z]),
        handedness: handedness[i]?.[0]?.categoryName ?? "Hand",
      })),
    });
  }
};
