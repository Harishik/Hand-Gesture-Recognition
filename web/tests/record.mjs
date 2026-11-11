// Records the running app to PNG frames for README GIFs, using the demo clip in place of a webcam.
//   python scripts/serve_web.py &   node web/tests/record.mjs <out-dir> <path> <seconds> [width] [height] [warmup-ms] [script]
// <script> is optional JavaScript evaluated in the page once recording starts (e.g. to click through slides).
// Frames come from Chrome's screencast with their timestamps; scripts/make_gif.py turns them into a GIF.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [out, path, seconds = "8", width = "1280", height = "720", warmup = "24000", script = ""] = process.argv.slice(2);
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = process.env.BASE || "http://localhost:8765";
const PORT = 9334;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(out, { recursive: true });

const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "gs-rec-"))}`,
  "--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required",
  `--window-size=${width},${height}`, "--hide-scrollbars", "about:blank",
]);

let ws;
for (let i = 0; i < 50 && !ws; i++) {
  try {
    const pages = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
    ws = new WebSocket(pages.find((p) => p.type === "page").webSocketDebuggerUrl);
  } catch { await sleep(200); }
}
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map();
const frames = [];
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  if (m.method === "Page.screencastFrame") {
    frames.push({ t: m.params.metadata.timestamp, data: m.params.data });
    send("Page.screencastFrameAck", { sessionId: m.params.sessionId });
  }
});
const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });

await send("Emulation.setDeviceMetricsOverride", { width: +width, height: +height, deviceScaleFactor: 1, mobile: +width < 600 });
await send("Page.navigate", { url: `${BASE}${path}` });
await sleep(+warmup);
await send("Page.startScreencast", { format: "png", everyNthFrame: 1, maxWidth: +width, maxHeight: +height });
if (script) send("Runtime.evaluate", { expression: script, awaitPromise: true });
await sleep(+seconds * 1000);
await send("Page.stopScreencast");
// Some pages (a lone 2D canvas animation) don't drive the screencast; fall back to timed screenshots.
if (frames.length < +seconds * 5) {
  frames.length = 0;
  const start = performance.now();
  while (performance.now() - start < +seconds * 1000) {
    const t = (performance.now() - start) / 1000;
    const { data } = await send("Page.captureScreenshot", { format: "png" });
    frames.push({ t, data });
  }
}
const t0 = frames[0]?.t ?? 0;
frames.forEach((f, i) => writeFileSync(join(out, `${String(i).padStart(4, "0")}_${Math.round((f.t - t0) * 1000)}.png`), Buffer.from(f.data, "base64")));
console.log(`wrote ${frames.length} frames over ${seconds}s to ${out}`);
chrome.kill();
process.exit(0);
