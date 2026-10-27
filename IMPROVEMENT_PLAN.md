# Hand Gesture Recognition: 2026 Upgrade Plan

## 1. Where the project stands today

**What's there:** a single `main.py` (~95 lines) plus a hand-made `gestures/` folder.

**How it works:**
1. `cvzone.HandDetector` (a wrapper over the legacy MediaPipe Hands API) finds hand bounding boxes in the webcam feed.
2. Each hand crop is converted to grayscale, resized to 50×50 and compared by **SSIM** (structural similarity) against all 34 reference photos.
3. The label of the most similar photo is drawn on screen, together with the SSIM score shown as "accuracy %".

**The dataset:** 34 JPEGs in 6 classes, all taken by one person against one background in a single session (1 June 2024).

| Class | fist | hello | okay | peace | thumbs_down | thumbs_up |
|---|---|---|---|---|---|---|
| Images | 6 | 4 | 5 | 12 | 3 | 4 |

### Problems a reviewer would spot

**Correctness bugs**
- **Hard-coded absolute paths.** The code points to `C:/Users/singh/PycharmProjects/...`, but the project now lives in `Downloads/`. As it stands, the script crashes on startup on any machine, including yours.
- **The crop includes cvzone's drawings.** `detector.findHands(frame)` draws the landmark skeleton and bounding box onto `frame` in place, and the crop is then taken from that same frame. Live crops have coloured lines over them; the reference photos don't.
- **Negative coordinates.** The bbox can go negative near the image edge, and `frame[y_min:y_max, ...]` then silently wraps around.
- **The "accuracy %" isn't accuracy.** It's the raw SSIM similarity. A reviewer will catch this right away.
- **No way to say "no gesture".** The argmax always returns one of the 6 labels, so any hand, or a face caught in the crop, gets classified.

**Method limitations**
- SSIM on 50×50 grayscale pixels isn't learning; it's template matching. It breaks under changes in lighting, background, skin tone, scale, rotation, left vs. right hand and camera.
- 34 images from 1 subject can't support any claim about generalisation, and there's no train/test split or measured metric.
- `cvzone` wraps `mp.solutions`, the legacy MediaPipe API. Recent `mediapipe` releases (≈0.10.30+) removed it, so a fresh `pip install` breaks the code. Only the pinned 0.10.14 in the old venv still works.

**Repo hygiene**
- No README, `requirements.txt`/`pyproject.toml`, licence, tests or `.gitignore`. The folder is not a git repo.
- `.venv/` (~810 MB) and `.idea/` sit inside the project folder and must never be committed.

**Verdict:** this reads as a 2024 tutorial-level project. The upgrade below turns it into a measured, reproducible ML system with a live demo.

---

## 2. Target: what the 2026 version should demonstrate

A recruiter or ML engineer skimming the repo should see:

1. **Real data at scale**: training on HaGRIDv2 (1M+ images, ~38k subjects), with a subject-disjoint split.
2. **A modern two-stage architecture**: hand detection and 21-point landmarks, then a learned classifier. An image-based model serves as a comparison baseline.
3. **Rigorous evaluation**: macro-F1, confusion matrix, calibration, open-set rejection, per-demographic slices and latency on CPU.
4. **Edge deployment**: ONNX/INT8, real-time on a laptop CPU, plus a **browser demo** anyone can try from the README.
5. **A distinctive feature**: few-shot "teach a new gesture in 30 seconds" and/or dynamic (motion) gestures.
6. **Engineering quality**: packaged code, configs, tests, CI, a model card and reproducible training.

---

## 3. Datasets (replacing and augmenting the hand-made one)

### Primary: HaGRIDv2 (recommended)
- **Repo:** https://github.com/hukenovs/hagrid · **Paper:** arXiv [2412.01508](https://arxiv.org/abs/2412.01508)
- **Contents:** 1,086,158 FullHD images in 33 gesture classes plus `no_gesture`.
- **Annotations:** bounding boxes, **MediaPipe 21-point hand landmarks for every hand**, `user_id`, and age/gender/race metadata.
- **Split:** 76/9/15 train/val/test, already split **by subject**.
- **All 6 of your classes exist under different names:**

| Your class | HaGRID class |
|---|---|
| hello | `palm` |
| fist | `fist` |
| peace | `peace` |
| okay | `ok` |
| thumbs_up | `like` |
| thumbs_down | `dislike` |

- **Practical tip:** you don't need the 1.5 TB of images for the main model. The **annotations archive alone** contains landmarks for every hand, so the landmark classifier trains on ~1M samples from a small download. Download images only for the classes you need, or use the 512 px version (119 GB total), for the image-based baseline and the YOLO detector.
- **Licence:** a variant of **CC BY-SA 4.0**; read their [specific licence PDF](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf). Attribute HaGRID in the README and model card. ShareAlike can apply to redistributed data or derivatives, so don't re-upload their images; publish your own code and weights with the licence noted.

### Secondary / optional
| Dataset | Use | Notes |
|---|---|---|
| **Ultralytics Hand-Keypoints** (26,768 images, 21 kpts) | Train a YOLO26-pose hand-landmark model as an alternative to MediaPipe | Ready YOLO format, ~369 MB; [docs](https://docs.ultralytics.com/datasets/pose/hand-keypoints) |
| **IPN Hand** (4k+ gestures, 800k frames, 50 subjects, 13 classes) | Dynamic/continuous gestures (swipe, click, zoom) | [GitHub](https://github.com/GibranBenitez/IPN-hand); check the licence in the repo |
| **Jester** (148k clips, 27 classes) | Large-scale dynamic gestures | Registration required; reported as **CC BY-NC-ND 4.0**, so fine for research or a portfolio but not commercial use |
| **FreiHAND / InterHand2.6M** | Only if you go to 3D hand pose | Optional stretch |

### What to do with your hand-made images
Don't discard them; **repurpose them as an out-of-distribution "personal test set"**. Reporting "HaGRID test macro-F1 = X, and on a completely unseen user/camera = Y" is a far stronger portfolio story than training on them.

Extend it with a small `scripts/capture.py` tool:
- record ~50 frames per class across 2–3 lighting setups, both hands, plus `no_gesture`;
- ideally include 2–3 friends, with their consent;
- store landmarks as JSON alongside the images.

---

## 4. Proposed architecture

```
Webcam frame
   │
   ▼
[Stage 1] Hand detection + 21 landmarks
   ├─ A: MediaPipe Tasks HandLandmarker (.task model, Tasks API — not legacy mp.solutions)
   └─ B: YOLO26n-pose fine-tuned on Hand-Keypoints (comparison / fully-owned model)
   │
   ▼
[Stage 2] Feature engineering (per hand)
   wrist-origin translation · scale-normalise by palm size · mirror left→right hand
   + finger joint angles + fingertip pairwise distances  (~100-dim vector)
   │
   ▼
[Stage 3] Gesture classifier (PyTorch)
   MLP (main model) vs. small Transformer over 21 keypoint tokens (ablation)
   trained on HaGRIDv2 landmarks; 6 target classes + no_gesture (or all 34)
   │
   ▼
[Stage 4] Temporal layer
   EMA smoothing / majority vote over N frames + confidence threshold → "unknown"
   (optional) GRU/TCN over landmark sequences for dynamic gestures (IPN Hand)
   │
   ▼
Action layer (demo app): media play/pause, volume, slide next/prev, screenshot…
```

**Baselines to compare against** (an ablation table makes the README):

| # | Model | Input |
|---|---|---|
| 0 | Your original SSIM template matching | 50×50 grayscale crop |
| 1 | Landmark MLP | 21×2 normalised keypoints |
| 2 | Landmark Transformer | keypoint tokens |
| 3 | Image classifier: fine-tuned ConvNeXt-V2-Tiny / EfficientNet, or a DINOv2/v3 linear probe | 224 px hand crop |
| 4 | One-stage YOLO26 gesture detector trained on HaGRID boxes | full frame |
| 5 | Google's stock MediaPipe Gesture Recognizer (+ Model Maker custom head) | reference point |

Report for each: **macro-F1, accuracy, per-class F1, ECE (calibration), CPU latency (ms/frame), model size (MB)** on the HaGRID test split *and* your personal test set.

---

## 5. Evaluation protocol (what makes it "highly technical")

- **Subject-disjoint splits** using HaGRID's `user_id`. Never let the same person appear in train and test.
- **Open-set rejection:** train with `no_gesture` and tune a softmax threshold. Report a precision–recall curve for "gesture vs. no gesture" (the false-trigger rate matters most in real HCI).
- **Calibration:** reliability diagram plus ECE, with temperature scaling.
- **Robustness:** test-time perturbations (brightness, blur, rotation ±30°, left hand, partial occlusion).
- **Fairness slices:** per age, gender and skin-tone group, using HaGRID's `meta` field. Reviewers rarely see this in portfolio projects.
- **Latency:** frames per second end-to-end on your laptop CPU, PyTorch FP32 vs. ONNX Runtime FP32 vs. INT8.
- **Error analysis:** a gallery of the worst confusions (e.g. `peace` vs. `two_up`, `like` vs. `fist`).

---

## 6. Standout features (pick 1–2)

1. **Few-shot custom gestures:** press a key, perform a new gesture for 30 frames, and the app adds a class. This could use prototypes (nearest class mean) in the learned embedding space of the landmark model, or a quick logistic-regression head. It's very demo-able.
2. **Dynamic gestures:** swipe left/right, pinch-zoom and click, from landmark sequences with a GRU/TCN on IPN Hand, or HaGRID's own [dynamic_gestures](https://github.com/ai-forever/dynamic_gestures) approach as a reference.
3. **Gesture-controlled app:** control Spotify/YouTube/PowerPoint with gestures, with a debounced state machine so actions don't fire repeatedly.
4. **In-browser demo:** MediaPipe Tasks for Web plus your classifier exported to ONNX and run with ONNX Runtime Web. Host it on GitHub Pages or Hugging Face Spaces. "Try it live" links get clicked; repos without them often don't.

---

## 7. Repo structure & engineering

```
hand-gesture-recognition/
├── pyproject.toml            # uv-managed, Python 3.12, pinned deps
├── README.md                 # demo GIF, results table, architecture diagram, quickstart
├── MODEL_CARD.md / DATA.md   # intended use, limitations, licences, attributions
├── configs/                  # YAML (or Hydra) for data/model/train
├── src/hgr/
│   ├── data/                 # HaGRID annotation parsing, landmark features, splits
│   ├── models/               # mlp.py, transformer.py, image_baseline.py
│   ├── train.py  evaluate.py  export_onnx.py
│   └── app/                  # realtime.py (OpenCV), actions.py, fewshot.py
├── scripts/                  # download_hagrid.py, capture.py, benchmark_latency.py
├── web/                      # browser demo (MediaPipe JS + onnxruntime-web)
├── tests/                    # feature normalisation, model I/O shapes, ONNX parity
├── notebooks/                # EDA + error analysis (light, outputs cleared)
└── .github/workflows/ci.yml  # ruff + pytest on push
```

- **Tooling:** `uv`, `ruff` (lint + format), `pytest`, `pre-commit`, GitHub Actions CI.
- **Experiment tracking:** MLflow, W&B or Hugging Face Trackio. Link one public run in the README.
- **Model weights:** publish to the Hugging Face Hub (you're already logged in there as `Harishik`) with a model card, not in git.
- **Docker:** an optional `Dockerfile` for the training/eval environment.
- **`.gitignore`:** `.venv/`, `.idea/`, `data/`, `runs/`, `*.onnx`, `*.pt`.
- **Licence:** MIT for your code. Note the HaGRID licence separately for data-derived artefacts.

---

## 8. Phased roadmap

| Phase | Deliverable | Done when |
|---|---|---|
| **0. Rescue** | Fix paths, clean crop, clamp bbox, rename "accuracy" → "similarity", git init, `.gitignore`, keep as `legacy/` baseline | Old version runs from a clean clone |
| **1. Data** | HaGRIDv2 annotation loader, 6-class + `no_gesture` subset, landmark feature pipeline, EDA notebook | Train/val/test tensors cached; class balance plotted |
| **2. Core model** | Landmark MLP + Transformer, training/eval scripts, metrics | Test macro-F1 reported; beats SSIM baseline on your personal set |
| **3. Real-time app** | MediaPipe Tasks HandLandmarker + classifier + temporal smoothing + rejection | ≥30 FPS on laptop CPU, no flicker, "unknown" when idle |
| **4. Benchmarks** | Image baseline (ConvNeXt/DINO) + YOLO26 detector + stock MediaPipe comparison, robustness & fairness slices | Ablation table + plots in README |
| **5. Deploy** | ONNX + INT8, latency benchmark, web demo on GH Pages / HF Space, weights on HF Hub | Public "Try it" link works on a phone |
| **6. Standout** | Few-shot custom gestures and/or dynamic gestures + gesture-controlled app | Demo GIF/video in README |
| **7. Polish** | Tests, CI, model card, architecture diagram, short write-up/blog post | CI green; README tells the full story in 60 seconds |

Phases 0–3 already give a solid, modern project. Phases 4–6 are what make it stand out in 2026.

---

## 9. Sources
- HaGRID / HaGRIDv2 repo and licence: https://github.com/hukenovs/hagrid
- HaGRIDv2 paper: https://arxiv.org/abs/2412.01508
- HaGRID dynamic gestures: https://github.com/ai-forever/dynamic_gestures
- MediaPipe Gesture Recognizer and Model Maker customisation: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer · https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer
- MediaPipe `solutions` module removal report: https://github.com/google-ai-edge/mediapipe/issues/6206
- Ultralytics Hand-Keypoints dataset (YOLO26-pose): https://docs.ultralytics.com/datasets/pose/hand-keypoints
- IPN Hand: https://github.com/GibranBenitez/IPN-hand · arXiv 2005.02134
- Jester: https://www.qualcomm.com/developer/software/jester-dataset
