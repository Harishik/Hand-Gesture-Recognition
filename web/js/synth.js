// A small generative ambient track, synthesised live with Web Audio: no audio files, nothing downloaded.
// Pad chords through a slowly breathing low-pass filter, a soft kick and an off-beat shaker, at 76 BPM.
// The analyser output drives the player's equaliser, so the bars show the real spectrum.

const BPM = 76;
const BEAT = 60 / BPM;
// Am9 -> Fmaj7 -> Cmaj9 -> G6, two bars each (MIDI note numbers).
const CHORDS = [
  [45, 52, 55, 59, 60],
  [41, 48, 52, 55, 57],
  [48, 55, 59, 62, 64],
  [43, 50, 55, 59, 64],
];
const hz = (midi) => 440 * 2 ** ((midi - 69) / 12);

export class AmbientSynth {
  constructor() {
    this.ctx = null;
    this.started = false;
    this.playing = false;
    this.volume = 0.6;
    this.muted = false;
    // Browsers only let audio start from a real click or key press. Create the context on the first one anywhere
    // (e.g. "Turn on camera"), so a later gesture-triggered Play is allowed to make sound.
    const unlock = () => this.ensure()?.resume();
    addEventListener("pointerdown", unlock, { once: true, capture: true });
    addEventListener("keydown", unlock, { once: true, capture: true });
  }

  get available() {
    return typeof AudioContext !== "undefined";
  }

  /** True if the browser will actually output sound right now. */
  get unlocked() {
    return this.ctx?.state === "running";
  }

  ensure() {
    if (this.ctx || !this.available) return this.ctx;
    const ctx = (this.ctx = new AudioContext());
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.82;
    this.master.connect(this.analyser).connect(ctx.destination);
    this.bins = new Uint8Array(this.analyser.frequencyBinCount);
    return ctx;
  }

  start() {
    const ctx = this.ensure();
    if (!ctx || this.started) return;
    this.started = true;

    // Pad: two detuned saws per chord note, through a low-pass filter whose cutoff breathes with a slow LFO.
    this.filter = ctx.createBiquadFilter();
    this.filter.type = "lowpass";
    this.filter.frequency.value = 900;
    this.filter.Q.value = 0.7;
    const lfo = ctx.createOscillator();
    const lfoDepth = ctx.createGain();
    lfo.frequency.value = 0.07;
    lfoDepth.gain.value = 450;
    lfo.connect(lfoDepth).connect(this.filter.frequency);
    lfo.start();
    this.padGain = ctx.createGain();
    this.padGain.gain.value = 0.05;
    this.filter.connect(this.padGain).connect(this.master);
    this.voices = CHORDS[0].map((note) =>
      [-6, 6].map((cents) => {
        const o = ctx.createOscillator();
        o.type = "sawtooth";
        o.frequency.value = hz(note);
        o.detune.value = cents;
        o.connect(this.filter);
        o.start();
        return o;
      }),
    );

    // Percussion bus.
    this.drums = ctx.createGain();
    this.drums.gain.value = 0.55;
    this.drums.connect(this.master);
    const noise = ctx.createBuffer(1, ctx.sampleRate * 0.2, ctx.sampleRate);
    const data = noise.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    this.noise = noise;

    this.beat = 0;
    this.nextTime = ctx.currentTime + 0.1;
    this.timer = setInterval(() => this.schedule(), 50);
  }

  /** Look-ahead scheduler: queues the next ~150 ms of events on the audio clock. */
  schedule() {
    const ctx = this.ctx;
    while (this.nextTime < ctx.currentTime + 0.15) {
      const t = this.nextTime;
      const bar = Math.floor(this.beat / 4);
      if (this.beat % 8 === 0) this.setChord(CHORDS[Math.floor(bar / 2) % CHORDS.length], t);
      if (this.beat % 4 === 0 || this.beat % 4 === 2) this.kick(t, this.beat % 4 === 0 ? 1 : 0.6);
      this.shaker(t + BEAT / 2);
      this.beat += 1;
      this.nextTime += BEAT;
    }
  }

  setChord(notes, t) {
    this.voices.forEach((pair, i) => pair.forEach((o) => o.frequency.setTargetAtTime(hz(notes[i]), t, 0.25)));
  }

  kick(t, level) {
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.frequency.setValueAtTime(110, t);
    o.frequency.exponentialRampToValueAtTime(42, t + 0.18);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.5 * level, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
    o.connect(g).connect(this.drums);
    o.start(t);
    o.stop(t + 0.4);
  }

  shaker(t) {
    const ctx = this.ctx;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const hp = ctx.createBiquadFilter();
    hp.type = "highpass";
    hp.frequency.value = 6500;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.08, t + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.09);
    src.connect(hp).connect(g).connect(this.drums);
    src.start(t);
    src.stop(t + 0.12);
  }

  apply() {
    if (!this.ctx) return;
    const target = this.playing && !this.muted ? this.volume * 0.9 : 0;
    this.master.gain.setTargetAtTime(target, this.ctx.currentTime, 0.12); // ~0.4 s fade, never a click
  }

  /** Returns false if the browser is still holding audio back (no click on the page yet). */
  play() {
    if (!this.available) return false;
    this.start();
    this.ctx.resume();
    this.playing = true;
    this.apply();
    return this.unlocked;
  }

  pause() {
    this.playing = false;
    this.apply();
  }

  setVolume(v) {
    this.volume = v;
    this.apply();
  }

  setMuted(m) {
    this.muted = m;
    this.apply();
  }

  /** Fills `out` (length n) with 0..1 levels from low to high frequency, on a log scale. */
  spectrum(out) {
    if (!this.analyser) return out.fill(0);
    this.analyser.getByteFrequencyData(this.bins);
    const n = out.length;
    const max = this.bins.length;
    const top = max * 0.4; // the music lives below ~9 kHz; above that is empty and would show as dead bars
    for (let i = 0; i < n; i++) {
      const a = Math.floor(2 * (top / 2) ** (i / n));
      const b = Math.max(a + 1, Math.floor(2 * (top / 2) ** ((i + 1) / n)));
      let s = 0;
      for (let k = a; k < b && k < max; k++) s = Math.max(s, this.bins[k]);
      out[i] = s / 255;
    }
    return out;
  }
}
