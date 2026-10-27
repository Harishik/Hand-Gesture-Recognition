# Gesture Studio: real-time hand gesture recognition

A real-time hand gesture system that runs entirely in the browser, trained on 1.3M hands from HaGRIDv2:
- **33 gestures:** 26 one-hand and 7 two-hand.
- **Swipes and an air cursor:** swipe left, right, up or down; point and pinch to click.
- **Custom gestures:** teach a new one from a 3-second recording.
- **3D hand view:** your hand rendered live in 3D, driving a gesture-controlled object.

Everything runs on-device; no video leaves the browser.

```bash
python scripts/serve_web.py      # then open http://localhost:8765 and press "Start camera"
```

## What it does

| Feature | How |
|---|---|
| **33 static gestures** | MediaPipe HandLandmarker → 67 landmark features → MLP (26 one-hand classes + rejection) |
| **Two-hand gestures** (heart, prayer, X sign, timeout, frame…) | A second MLP on both hands jointly (221 features: both hands plus their relative pose) |
| **Swipes** | Palm-centre velocity over 320 ms, normalised by palm size, so it works at any distance |
| **Air cursor** | Index fingertip with a One Euro filter; pinch to click any control, pinch-drag sliders |
| **Dwell to confirm** | A gesture must be held for 450 ms to fire, once per hold; avoids accidental triggers |
| **Teach a gesture** | Few-shot nearest neighbour in the MLP's learned embedding space; saved in the browser |
| **3D lab** (Three.js) | Live 3D hand skeleton from MediaPipe world landmarks. Fist and move to rotate an object, two open hands to scale, pinch to change material |
| **Gesture → action bindings** | Every gesture can drive a demo control deck (slides, media player, reactions, snapshots), remappable in the UI |
| **Live diagnostics** | Top-5 probabilities per hand, FPS, landmark and classifier latency, event log |

The design choices (pinch-to-click, immediate visual feedback, dwell confirmation, jitter filtering, on-device processing) follow current XR hand-tracking guidance. See [docs/RESEARCH.md](docs/RESEARCH.md) for the survey of papers, products and open-source projects behind them.

## Results

All test numbers are on **HaGRIDv2 test subjects never seen in training**, with subject-disjoint splits (see [docs/DATA.md](docs/DATA.md)).

| Model | Test set | Accuracy | Macro precision |
|---|---|---|---|
| One-hand, 26 gestures | 26 gestures + no gesture (153,755 hands) | **98.8%** | **98.9%** |
| | Open-set: also rejects halves of two-hand gestures (201,133 hands) | 94.7% | 95.7% |
| Two-hand, 7 gestures | Pairs of hands (44,195 pairs) | **99.3%** | **98.0%** |
| One-hand, original 6 gestures | 6 + no gesture (60,488 hands) | 99.4% | 99.2% |

- **Fairness:** gesture recall stays within 98.1–99.0% across the subjects' auto-annotated gender and race groups.
- **Calibration:** expected calibration error is 0.014.
- **Speed:** the classifiers run in about 1 ms. End to end, the Python pipeline reaches about 21 FPS on a laptop CPU. In the browser, the hand landmarker uses the GPU (WebGL) when available.

The method, ablations and caveats are in [docs/MODEL.md](docs/MODEL.md). Key findings:
- **Open-set training matters:** training without "other gesture" negatives drops live precision to 32%.
- **HaGRID's "inverted" classes are back-of-hand views, not upside-down hands.**
- **HaGRID's official split leaks two subjects between train and test.**
- **HaGRID mixes image aspect ratios** that its annotations don't record.

## Architecture

```
camera ─► MediaPipe HandLandmarker (WASM/WebGL) ─► 21 landmarks per hand (2D + 3D world)
            │
            ├─► features (67) ─► one-hand MLP ─► temperature + rejection bias ─► EMA smoothing ─┐
            ├─► pair features (221) ─► two-hand MLP ───────────────────────────────────────────┤
            ├─► palm trajectory ─► swipes                                                       ├─► dwell ─► events ─► actions
            ├─► thumb–index distance ─► pinch ─► One Euro cursor ─► click / drag                ┘
            └─► world landmarks ─► Three.js 3D lab
```

The models are trained in PyTorch, then exported with BatchNorm folded into the linear layers. In the browser they run in a small, dependency-free JavaScript module. A parity test checks that the JavaScript features and probabilities match Python.

## Repository layout

```
├── web/                   # the browser app (no build step)
│   ├── index.html, css/
│   ├── js/                # app, engine (dwell/swipe/pinch), features, mlp, scene3d, custom, filters
│   ├── models/            # exported classifiers (hand.json, pair.json) + parity fixture
│   └── tests/parity.mjs   # JS vs Python check
├── src/hgr/               # Python: data, features, training, evaluation, real-time pipeline
│   ├── hagrid.py, splits.py, features.py, augment.py, pairs.py, labels.py
│   ├── model.py, train.py, train_pairs.py, evaluate.py, metrics.py
│   ├── landmarker.py, pipeline.py
│   └── legacy/            # the original 2024 SSIM template-matching version
├── scripts/               # download, cache, export, serve, live OpenCV demo, clip recorder/evaluator
├── docs/                  # DATA.md, MODEL.md, RESEARCH.md, figures
├── reports/               # evaluation outputs (JSON)
├── data/personal/         # 34 hand-captured photos used as an extra out-of-distribution check
└── tests/
```

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"
python scripts/download_hagrid.py                      # ~720 MB annotations (with landmarks)
python scripts/build_hagrid_cache.py                   # 1.31M hands -> data/processed (~30 s)
python -m hgr.train --classes hagrid --out models/gesture_mlp_hagrid.pt   # ~25 min on a laptop CPU
python -m hgr.train_pairs                              # two-hand model, ~6 min
python -m hgr.evaluate --model models/gesture_mlp_hagrid.pt --out reports/metrics_hagrid.json
python scripts/export_web.py                           # -> web/models
pytest                                                 # includes the JS/Python parity test (needs Node)
```

## From the 2024 version

The original class project compared 50×50 grayscale crops against 34 photos of one person using SSIM (structural similarity). It recognised 6 gestures, had no way to reject other hand shapes, and its on-screen "accuracy" was really a similarity score. That version is kept in `src/hgr/legacy/` as a baseline.

## Acknowledgements

- **Data:** HaGRIDv2, by Nuzhdin et al. ([arXiv:2412.01508](https://arxiv.org/abs/2412.01508)), licensed under a variant of CC BY-SA 4.0 ([terms](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)).
- **Hand tracking:** [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) by Google.
- **3D:** [Three.js](https://threejs.org).
