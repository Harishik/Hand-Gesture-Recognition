// Small UI pieces shared by both pages: transient notices and the keyboard-shortcut sheet.

/** Shows a transient notice. tone: "live" | "busy" | "err". Returns a function that dismisses it. */
export function notice(text, { tone = "busy", action, timeout = 6000 } = {}) {
  const host = document.getElementById("notices");
  const el = document.createElement("div");
  el.className = "notice";
  el.innerHTML = `<span class="dot ${tone}"></span><span></span>`;
  el.lastChild.textContent = text;
  const dismiss = () => {
    if (el.classList.contains("out")) return;
    el.classList.add("out");
    setTimeout(() => el.remove(), 200);
  };
  if (action) {
    const b = document.createElement("button");
    b.className = "btn ghost small";
    b.textContent = action.label;
    b.onclick = () => { action.run(); dismiss(); };
    el.append(b);
  }
  while (host.children.length > 2) host.firstChild.remove();
  host.append(el);
  if (timeout) setTimeout(dismiss, timeout);
  return dismiss;
}

/** Wires the "?" button and dialog. Returns a toggle function for the keyboard handler. */
export function bindHelp(dialog, button) {
  button?.addEventListener("click", () => dialog.showModal());
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog || e.target.closest("[data-close]")) dialog.close();
  });
  return () => (dialog.open ? dialog.close() : dialog.showModal());
}

/** Screen-reader announcement of a confirmed event (the visual panels update far too often to be live regions). */
export function announce(text) {
  const el = document.getElementById("announce");
  if (!el) return;
  el.textContent = "";
  requestAnimationFrame(() => (el.textContent = text));
}

const SOURCE_KEY = "gs.source";
/** Remembers the running source for this tab ("camera" or a clip URL), so the other view can resume it. */
export const session = {
  remember(source) {
    try { sessionStorage.setItem(SOURCE_KEY, source); } catch { /* storage unavailable */ }
  },
  forget() {
    try { sessionStorage.removeItem(SOURCE_KEY); } catch { /* storage unavailable */ }
  },
  /** The source to resume, or null. A camera is only resumed if permission is already granted (no surprise prompt). */
  async resumable() {
    let source = null;
    try { source = sessionStorage.getItem(SOURCE_KEY); } catch { return null; }
    if (source !== "camera") return source;
    try {
      const p = await navigator.permissions?.query({ name: "camera" });
      return p?.state === "granted" ? "camera" : null;
    } catch {
      return null;
    }
  },
};

/** Fills the onboarding problem box; problem = { title, body, openElsewhere } or null to hide it. */
export function showProblem(problem) {
  const box = document.getElementById("problem");
  box.hidden = !problem;
  if (!problem) return;
  document.getElementById("problem-title").textContent = problem.title;
  document.getElementById("problem-body").textContent = problem.body;
  const actions = document.getElementById("problem-actions");
  actions.hidden = !problem.openElsewhere;
  const url = location.href.split("?")[0];
  document.getElementById("open-elsewhere").onclick = () => window.open(url, "_blank", "noopener");
  document.getElementById("copy-link").onclick = async (e) => {
    try {
      await navigator.clipboard.writeText(url);
      e.target.textContent = "Link copied";
    } catch {
      e.target.textContent = url;
    }
  };
}

/** True when a keyboard shortcut should be ignored because the user is typing or using a control. */
export const typing = (e) =>
  e.metaKey || e.ctrlKey || e.altKey || Boolean(e.target.closest?.("input, textarea, select, [contenteditable]"));
