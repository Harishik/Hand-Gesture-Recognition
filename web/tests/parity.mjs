// Checks the JavaScript feature extraction and MLP against Python outputs in web/models/parity.json.
// Usage: node web/tests/parity.mjs   (exit code 1 on mismatch)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { featurize, pairFeatures } from "../js/features.js";
import { GestureModel } from "../js/mlp.js";

const models = join(dirname(fileURLToPath(import.meta.url)), "..", "models");
const load = (name) => JSON.parse(readFileSync(join(models, name), "utf8"));
const fixture = load("parity.json");
const hand = new GestureModel(load("hand.json"));
const pair = new GestureModel(load("pair.json"));

const maxDiff = (a, b) => Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i])));
let worstFeature = 0;
let worstProb = 0;
let argmaxMismatches = 0;
const argmax = (v) => v.indexOf(Math.max(...v));

for (const s of fixture.hands) {
  const f = featurize(s.landmarks, s.aspect);
  const p = hand.predictProba(f);
  worstFeature = Math.max(worstFeature, maxDiff(f, s.features));
  worstProb = Math.max(worstProb, maxDiff(p, s.probs));
  if (argmax(p) !== argmax(s.probs)) argmaxMismatches++;
}
for (const s of fixture.pairs) {
  const f = pairFeatures(s.landmarks[0], s.landmarks[1], s.aspect);
  const p = pair.predictProba(f);
  worstFeature = Math.max(worstFeature, maxDiff(f, s.features));
  worstProb = Math.max(worstProb, maxDiff(p, s.probs));
  if (argmax(p) !== argmax(s.probs)) argmaxMismatches++;
}

console.log(JSON.stringify({ worstFeature, worstProb, argmaxMismatches }));
process.exit(worstFeature < 1e-4 && worstProb < 1e-4 && argmaxMismatches === 0 ? 0 : 1);
