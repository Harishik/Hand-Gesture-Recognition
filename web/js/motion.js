// Frame-rate independent motion primitives shared by the studio and the classic dashboard.
// Everything here is driven by the render loop's dt (seconds), so it looks the same at 30, 60 or 144 Hz.

export const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

/** Exponential approach factor for a time constant `tau` (seconds): x += (target - x) * approach(dt, tau). */
export const approach = (dt, tau) => (reducedMotion.matches || tau <= 0 ? 1 : 1 - Math.exp(-dt / tau));

/** Damped spring on a scalar. `k` stiffness (1/s²), `zeta` damping ratio (1 = critical, < 1 = soft overshoot). */
export class Spring {
  constructor(value = 0, { k = 170, zeta = 0.8 } = {}) {
    this.value = value;
    this.target = value;
    this.velocity = 0;
    this.k = k;
    this.c = 2 * Math.sqrt(k) * zeta;
  }

  set(target) {
    this.target = target;
    return this;
  }

  jump(value) {
    this.value = this.target = value;
    this.velocity = 0;
    return this;
  }

  step(dt) {
    if (reducedMotion.matches) return this.jump(this.target).value;
    // Sub-step long frames so a stall never makes the spring explode.
    let left = Math.min(dt, 0.1);
    while (left > 0) {
      const h = Math.min(left, 1 / 120);
      const acc = this.k * (this.target - this.value) - this.c * this.velocity;
      this.velocity += acc * h;
      this.value += this.velocity * h;
      left -= h;
    }
    return this.value;
  }

  get settled() {
    return Math.abs(this.target - this.value) < 1e-3 && Math.abs(this.velocity) < 1e-3;
  }
}

/** A number shown in the DOM that eases towards its target instead of jumping. */
export class TweenNumber {
  constructor(el, { tau = 0.12, format = (v) => Math.round(v).toString() } = {}) {
    this.el = el;
    this.value = 0;
    this.target = 0;
    this.tau = tau;
    this.format = format;
    this.text = "";
  }

  set(v) {
    if (Number.isFinite(v)) this.target = v;
  }

  step(dt) {
    this.value += (this.target - this.value) * approach(dt, this.tau);
    const text = this.format(this.value);
    if (text !== this.text) this.el.textContent = this.text = text;
  }
}

/**
 * A keyed list whose rows glide to new positions (FLIP-style, but spring-driven every frame): rows that enter fade
 * and slide in, rows that leave fade out, and reorders slide instead of snapping. Rows are absolutely positioned
 * inside `container`; `render(key, el, data)` fills or updates a row's content.
 */
export class SpringList {
  constructor(container, { rowHeight, gap = 0, create, update, max = Infinity }) {
    this.container = container;
    this.rowHeight = rowHeight;
    this.gap = gap;
    this.create = create;
    this.update = update;
    this.max = max;
    this.rows = new Map();
    container.classList.add("spring-list");
  }

  /** items: [{ key, ...data }] in display order. */
  set(items) {
    const keep = new Set();
    items.slice(0, this.max).forEach((item, index) => {
      keep.add(item.key);
      let row = this.rows.get(item.key);
      if (!row) {
        const el = this.create(item);
        el.classList.add("spring-row");
        this.container.append(el);
        const y = new Spring(index * this.pitch + (this.rows.size ? -8 : 8), { k: 260, zeta: 0.86 });
        row = { el, y, alpha: new Spring(0, { k: 200, zeta: 1 }), leaving: false };
        this.rows.set(item.key, row);
      }
      row.leaving = false;
      row.index = index;
      row.y.set(index * this.pitch);
      row.alpha.set(1);
      row.el.removeAttribute("aria-hidden");
      this.update?.(row.el, item, index);
    });
    for (const [key, row] of this.rows) {
      if (keep.has(key) || row.leaving) continue;
      row.leaving = true;
      // Leaving rows get out of the way quickly; arriving ones take their time.
      row.alpha = new Spring(row.alpha.value, { k: 700, zeta: 1 }).set(0);
      row.y.set(row.y.target + 6);
      row.el.setAttribute("aria-hidden", "true");
    }
    // Grow instantly (rows must never spill over what follows), shrink smoothly once rows have faded.
    const h = Math.max(0, Math.min(items.length, this.max) * this.pitch - this.gap);
    const grow = h > (this.height ?? 0);
    this.container.style.transition = grow ? "none" : "";
    this.container.style.height = `${h}px`;
    this.height = h;
  }

  get pitch() {
    return this.rowHeight + this.gap;
  }

  step(dt) {
    for (const [key, row] of this.rows) {
      const y = row.y.step(dt);
      const a = Math.max(0, Math.min(1, row.alpha.step(dt)));
      row.el.style.transform = `translate3d(0, ${y.toFixed(2)}px, 0)`;
      row.el.style.opacity = a.toFixed(3);
      if (row.leaving && a < 0.01) {
        row.el.remove();
        this.rows.delete(key);
      }
    }
  }

  clear() {
    this.set([]);
  }
}

/**
 * Orders items ({ key, p }) by p, but keeps the previous order unless a later item beats an earlier one by more than
 * `margin`. Stops rows from trading places on noise.
 */
export function stableRank(items, previousKeys = [], margin = 0.05) {
  const prev = new Map(previousKeys.map((k, i) => [k, i]));
  return [...items].sort((a, b) => {
    if (Math.abs(a.p - b.p) <= margin && prev.has(a.key) && prev.has(b.key)) return prev.get(a.key) - prev.get(b.key);
    return b.p - a.p;
  });
}

/** One shared render clock for DOM motion. Subscribers get dt in seconds. */
const subscribers = new Set();
let last = performance.now();
function tick(now) {
  requestAnimationFrame(tick); // schedule first: one failing subscriber must never stop the clock
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  for (const fn of subscribers) {
    try {
      fn(dt, now);
    } catch (err) {
      console.error(err);
    }
  }
}
requestAnimationFrame(tick);
export const onFrame = (fn) => (subscribers.add(fn), () => subscribers.delete(fn));
