// Captures README screenshots of the running app, using the demo clip in place of a webcam.
//   python scripts/serve_web.py &   node web/tests/screenshot.mjs
// Drives a headless Chrome over the DevTools protocol (Node 22+, no dependencies).
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = process.env.BASE || "http://localhost:8765";
const OUT = new URL("../../docs/figures/", import.meta.url);
const PORT = 9333;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "gs-"))}`,
  "--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required",
  "--window-size=1440,860", "--hide-scrollbars", "about:blank",
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
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
});
const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });

await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 860, deviceScaleFactor: 1, mobile: false });
for (const [name, path] of [["studio", "/"], ["classic", "/classic.html"]]) {
  await send("Page.navigate", { url: `${BASE}${path}?video=test-media/demo.webm&autostart` });
  // Wait for a confident, steady prediction (>= 90% on two checks 1.5 s apart) so the ring has settled.
  const confident = async () => {
    const { result } = await send("Runtime.evaluate", { expression: "document.body.innerText" });
    const text = result.value || "";
    const pct = Math.max(0, ...[...text.matchAll(/(\d+)%/g)].slice(0, 2).map((m) => +m[1]));
    return !/No hand|No gesture|Starting camera/.test(text.slice(0, 600)) && pct >= 90;
  };
  for (let t = 0; t < 120; t++) {
    await sleep(500);
    if (t > 16 && (await confident())) { await sleep(1500); if (await confident()) break; }
  }
  const { data } = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(new URL(`screen_${name}.png`, OUT), Buffer.from(data, "base64"));
  console.log("wrote", name);
}
chrome.kill();
process.exit(0);
