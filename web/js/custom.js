// Few-shot custom gestures: record a couple of seconds of a new hand pose, then recognise it by nearest neighbour
// in the trained model's embedding space. Stored only in this browser (localStorage).

const STORAGE_KEY = "hgr.customGestures.v1";

const cosine = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

export class CustomGestures {
  constructor() {
    this.items = [];
    this.load();
  }

  load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      this.items = raw ? JSON.parse(raw).map((g) => ({ ...g, samples: g.samples.map((s) => Float32Array.from(s)) })) : [];
    } catch {
      this.items = [];
    }
  }

  save() {
    try {
      const data = this.items.map((g) => ({ ...g, samples: g.samples.map((s) => Array.from(s)) }));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch {
      // Storage unavailable (private mode): gestures still work for this session.
    }
  }

  /** Add a gesture from embeddings recorded while the user held the pose. */
  add(name, embeddings) {
    const id = `custom:${name.trim().toLowerCase().replace(/\s+/g, "_")}`;
    // Acceptance threshold from how consistent the recording was: mean within-sample similarity minus 3 std.
    const sims = [];
    for (let i = 0; i < embeddings.length; i++)
      for (let j = i + 1; j < embeddings.length; j++) sims.push(cosine(embeddings[i], embeddings[j]));
    const mean = sims.reduce((a, b) => a + b, 0) / Math.max(sims.length, 1);
    const std = Math.sqrt(sims.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(sims.length, 1));
    const threshold = Math.min(0.97, Math.max(0.8, mean - 3 * std));
    this.items = this.items.filter((g) => g.id !== id);
    this.items.push({ id, name: name.trim(), threshold, samples: embeddings });
    this.save();
    return this.items.at(-1);
  }

  remove(id) {
    this.items = this.items.filter((g) => g.id !== id);
    this.save();
  }

  /** Best matching custom gesture for an embedding, or null. */
  match(embedding) {
    let best = null;
    for (const g of this.items) {
      let sim = -1;
      for (const s of g.samples) sim = Math.max(sim, cosine(embedding, s));
      if (sim >= g.threshold && (!best || sim > best.similarity)) best = { id: g.id, name: g.name, similarity: sim };
    }
    return best;
  }
}
