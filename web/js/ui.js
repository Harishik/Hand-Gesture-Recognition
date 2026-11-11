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

/** True when a keyboard shortcut should be ignored because the user is typing or using a control. */
export const typing = (e) =>
  e.metaKey || e.ctrlKey || e.altKey || Boolean(e.target.closest?.("input, textarea, select, [contenteditable]"));
