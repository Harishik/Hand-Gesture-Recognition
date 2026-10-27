# Hand Gesture Recognition

Real-time hand gesture recognition from a webcam.

> **Status: under active rebuild.** The original 2024 version (SSIM template matching on 34 hand-captured photos) is preserved as a baseline. The new version is a learned, landmark-based pipeline trained on [HaGRIDv2](https://github.com/hukenovs/hagrid). See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the full roadmap.

## Gestures

| Class (HaGRIDv2 name) | Personal test-set folder |
|---|---|
| `palm` | `hello` |
| `fist` | `fist` |
| `peace` | `peace` |
| `ok` | `okay` |
| `like` | `thumbs_up` |
| `dislike` | `thumbs_down` |
| `no_gesture` | — |

## Repository layout

```
├── data/personal/         # 34 hand-captured photos: an out-of-distribution test set
├── docs/DATA.md           # dataset statistics, split audit, aspect-ratio analysis
├── docs/MODEL.md          # method, results, ablations, findings
├── reports/               # evaluation outputs (JSON)
├── scripts/               # download, cache building, landmark extraction, figures
├── src/hgr/
│   ├── hagrid.py          # HaGRIDv2 annotation loader and cache
│   ├── features.py        # landmark normalisation and feature vectors
│   ├── splits.py          # subject-disjoint split checks and splitting
│   ├── landmarker.py      # MediaPipe Tasks HandLandmarker wrapper
│   ├── augment.py         # landmark augmentation and aspect-ratio sampling
│   ├── model.py           # MLP + calibrated classifier checkpoint
│   ├── train.py, evaluate.py
│   ├── metrics.py         # accuracy, precision/recall/F1, calibration error
│   ├── pipeline.py        # real-time pipeline with temporal smoothing
│   ├── labels.py          # shared class vocabulary
│   └── legacy/            # original SSIM template-matching baseline
├── tests/
└── IMPROVEMENT_PLAN.md
```

## Results

| Test set | Accuracy | Precision |
|---|---|---|
| HaGRIDv2 test, unseen people (60k hands) | 99.4% | 99.2% |
| HaGRIDv2 open-set test (must also reject 26 other gestures, 201k hands) | 99.0% | 94.2% |
| Personal photos (unseen person, camera and background) | 93.9% | 100% |

It runs at about 21 FPS end to end on a laptop CPU. See [docs/MODEL.md](docs/MODEL.md) for the method, ablations, findings and caveats.

## Data

The dataset is HaGRIDv2: 1.31M annotated hands with MediaPipe landmarks, split by subject. See [docs/DATA.md](docs/DATA.md) for statistics and findings, including a train/test subject leak in the official split, and evidence that HaGRID mixes image aspect ratios, which the annotations don't record.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[legacy,dev]"
```

Train, evaluate and run the landmark classifier (needs the data cache from [docs/DATA.md](docs/DATA.md)):

```bash
python -m hgr.train                     # ~10 min on a laptop CPU -> models/gesture_mlp.pt
python -m hgr.evaluate                  # HaGRID test + personal photos -> reports/metrics.json
python scripts/live.py                  # live webcam demo
```

Measure real-time accuracy on your own recorded clips:

```bash
python scripts/record_clips.py --session yourname_room1   # guided: one 8 s clip per gesture
python scripts/eval_clips.py                              # frame-level metrics + latency -> reports/realtime_metrics.json
```

Run the legacy baseline (press `q` to quit):

```bash
python -m hgr.legacy.ssim_matcher
```

Run the tests:

```bash
pytest
```

## Baseline: SSIM template matching (2024)

Each detected hand is cropped, converted to 50×50 grayscale and compared with every reference photo using the structural similarity index (SSIM). The closest photo's label wins. The on-screen number is the SSIM score, **not** a confidence or an accuracy.

Known limitations, which are what motivate the rebuild:
- Only 34 reference images, from one person against one background.
- Sensitive to lighting, background, scale, rotation and which hand is used.
- No rejection class, so every hand gets one of the six labels.
- Depends on the legacy `mediapipe.solutions` API (pinned to `mediapipe==0.10.14`).

## Roadmap

1. **Data:** HaGRIDv2 landmark annotations, with subject-disjoint splits.
2. **Model:** a landmark MLP/Transformer classifier with `no_gesture` rejection.
3. **Real-time app:** MediaPipe Tasks HandLandmarker, temporal smoothing, 30+ FPS on CPU.
4. **Benchmarks:** image models, YOLO26, robustness and fairness slices.
5. **Deployment:** ONNX/INT8 and a browser demo.

## Acknowledgements

Training data: **HaGRIDv2**, by Nuzhdin et al., *HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition* ([arXiv:2412.01508](https://arxiv.org/abs/2412.01508)). It is licensed under a variant of CC BY-SA 4.0; see the [dataset licence](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf).
