# Model

## Pipeline

```
frame -> MediaPipe Tasks HandLandmarker (VIDEO mode, 21 landmarks per hand)
      -> features (67-dim: wrist-centred, palm-scaled points + 15 joint angles + 10 fingertip distances)
      -> MLP (3 x 256, BatchNorm, GELU, dropout 0.1) -> temperature-scaled softmax + no_gesture bias
      -> per-hand exponential moving average of probabilities (weight 0.6 on the previous average)
```

Classes: `palm`, `fist`, `peace`, `ok`, `like`, `dislike`, plus `no_gesture` (rejection).

## Training

- **Data:** HaGRIDv2 landmark cache, using the subject-disjoint train split (see [DATA.md](DATA.md)).
- **Label mapping:**
  - **Positives:** the six target gestures, plus `peace_inverted` → `peace`. HaGRID's "inverted" means the back of the hand faces the camera; see finding 2.
  - **Negatives (`no_gesture`):** HaGRID's `no_gesture` class plus all 26 other HaGRID gestures (open-set training).
- **Sampling:** 30,000 hands per class per epoch, drawn fresh each epoch, so the 845k negatives can't swamp the gestures.
- **Augmentation, regenerated every epoch:**
  - an image aspect ratio drawn from the measured HaGRID prior;
  - a left/right mirror;
  - ±20° rotation;
  - ±8% anisotropic stretch;
  - per-point jitter.
- **Optimisation:** AdamW with a one-cycle learning rate of 2e-3, 40 epochs, label smoothing 0.05, about 10 s per epoch on a laptop CPU.
- **Model selection:** the best epoch by class-balanced macro F1 on validation.
- **Calibration and rejection:** a softmax temperature is fitted on validation (T = 0.62). The `no_gesture` logit bias is tuned on validation to maximise class-balanced macro F1 (+0.10).
- **Clean test set:** the test split is never used for any choice.

## Results

Macro precision and recall average the six gestures, so "precision" answers this question: when the model names a gesture, how often is it right?

| Test set | n | Accuracy | Macro precision | Macro recall | Macro F1 | ECE |
|---|---|---|---|---|---|---|
| HaGRID test, core (six gestures + no_gesture) | 60,488 | **99.4%** | **99.2%** | 99.6% | 99.4% | 0.002 |
| HaGRID test, open-set (+ 26 other gestures to reject) | 201,133 | **99.0%** | **94.2%** | 99.6% | 96.8% | 0.002 |
| Personal photos (1 person, unseen camera/background) | 33 | **93.9%** | **100%** | 93.3% | 95.8% | — |

**Fairness:** gesture recall by the subject's auto-annotated demographics (HaGRID core test) stays within 98.9–99.7% for every group.

| Group | female | male | White | Black | Asian | Indian |
|---|---|---|---|---|---|---|
| Gesture recall | 99.7% | 99.5% | 99.6% | 99.5% | 99.3% | 99.2% |

**Real-time speed:** 47 ms per frame end to end, about **21 FPS** on a 12-thread laptop CPU. Hand landmarking takes about 40 ms of that; the classifier takes under 1 ms.

### Ablations

All were trained with the same recipe. These rows were evaluated before the `peace_inverted` alias existed, so their core test excludes that class.

| Variant | HaGRID core acc | Open-set precision | Personal acc |
|---|---|---|---|
| Gradient-boosting baseline (scikit-learn), negatives = no_gesture | 99.6% | — | 81.8% |
| MLP, negatives = no_gesture only | 99.5% | **32.4%** | 93.9% |
| MLP, negatives = all other gestures | 99.4% | 94.1% | 81.8% |
| MLP, negatives = all, ±45° rotation | 99.4% | 94.0% | 75.8% |
| **MLP, negatives = all, peace_inverted → peace (final)** | 99.4% | 94.2% | **93.9%** |

## Findings

1. **Training only on HaGRID's `no_gesture` class looks good offline, but it would misfire constantly live.**
   - That model reaches 99.5% on the core test.
   - On the open-set test, its gesture precision collapses to 32%: any other hand shape (`stop`, `one`, `two_up`…) triggers one of the six gestures.
   - Using every other HaGRID gesture as a negative keeps open-set precision at 94% with no loss on the core test.
2. **HaGRID's "inverted" classes are the back of the hand, not upside down.**
   - **Symptom:** four of the personal peace signs were rejected with 95–100% confidence.
   - **Evidence:** a nearest-neighbour lookup put all four squarely among HaGRID's `peace_inverted` hands (22–25 of their 25 neighbours).
   - **What it wasn't:** wider rotation augmentation (±45°) was tested first and didn't help.
   - **Fix:** mapping `peace_inverted` → `peace` fixed all four.
3. **Two remaining personal-set errors are landmarking failures, not classification errors.**
   - Both are fists photographed so close that the wrist is out of frame.
   - MediaPipe's skeleton for them is garbled, and their nearest HaGRID neighbours are `point` and `xsign`.
   - A classifier can't recover from bad landmarks; normal webcam framing avoids this.
4. **PyTorch's thread pool slowed MediaPipe down 2.5×.**
   - With both libraries in one process, hand landmarking went from 38 ms to 95 ms per frame.
   - Restricting PyTorch to one thread (the MLP needs no more) restored 39 ms, and the classifier still runs in under 1 ms.

## Caveats

- **The HaGRID numbers are optimistic for live use.** HaGRID's landmarks were produced by MediaPipe itself, and the 13% of hands it failed on were dropped. Its test set therefore measures the classifier on hands MediaPipe already handles well.
- **The personal set is no longer an untouched test.**
  - It's small: one photo is 3 percentage points.
  - Its errors led to the `peace_inverted` finding.
  - Final claims should therefore rest on recorded webcam clips from several people and rooms (`scripts/record_clips.py` and `scripts/eval_clips.py`).

## Full HaGRIDv2 vocabulary (33 gestures)

The system now recognises the full HaGRIDv2 gesture set with two models.

### One-hand model: 26 gestures + `no_gesture`

- **Training:** the same recipe as above with `--classes hagrid`, 30 epochs, 30,000 hands per class per epoch, about 25 minutes on a laptop CPU.
- **Labels:** the halves of two-handed gestures are negatives for this model, because the pair model handles them.
- **No aliases:** `peace_inverted` and the other `_inverted` classes are separate classes here. They count as their upright equivalents only in the "lenient" personal score.

| Test set (unseen subjects) | n | Accuracy | Macro precision | Macro recall | Macro F1 | ECE |
|---|---|---|---|---|---|---|
| Core: 26 gestures + no_gesture | 153,755 | **98.8%** | **98.9%** | 98.9% | 98.9% | 0.014 |
| Open-set: + halves of two-hand gestures | 201,133 | 94.7% | 95.7% | 98.8% | 96.8% | 0.021 |

- **Hardest classes:** the lowest per-class F1 scores are `one` 0.964, `mute` 0.965 and `point` 0.966. All three are index-finger poses that differ mainly in orientation.
- **Demographic slices:** gesture recall is between 98.1% and 99.0% for every group.
- **Personal photos:** 75.8% strict and 87.9% lenient.
  - **By definition:** four of the eight errors are back-of-hand V signs, now correctly called `peace_inverted` under HaGRID's own definition. The lenient score counts these as correct.
  - **Landmark failures:** the other four are close-up fists whose landmarks MediaPipe garbles (see finding 3 above).

### Two-hand model: 7 gestures + "two hands, no gesture"

- **Data:** images with exactly two landmark-annotated hands. The label is the two-handed gesture when HaGRID gives it to both hands; otherwise it's `no_gesture`, for example one gesture next to a relaxed hand.
- **Features (221):**
  - both hands' points in a shared frame, centred between the wrists and scaled by mean palm length;
  - each hand's own 67 features;
  - the vector and distance between the wrists.
  The features don't depend on which hand comes first. Augmentation applies the same transform to both hands.

| Test pairs (unseen subjects) | n | Accuracy | Macro precision | Macro recall |
|---|---|---|---|---|
| 7 two-hand gestures + no_gesture | 44,195 | **99.3%** | **98.0%** | 98.9% |

Per class precision / recall: `hand_heart` 0.991 / 0.999, `hand_heart2` 0.985 / 0.984, `holy` 0.968 / 0.990, `take_picture` 0.986 / 0.987, `thumb_index2` 0.995 / 0.999, `timeout` 0.941 / 0.966, `xsign` 0.994 / 0.997.

### Browser deployment

- **Export:** `scripts/export_web.py` folds each BatchNorm into its preceding Linear layer and writes the weights as base64 float32 (814 KB and 995 KB). The browser runs them in plain JavaScript with exact GELU.
- **Parity:** `tests/test_web_parity.py` runs `web/tests/parity.mjs` on real test hands. JavaScript and Python agree to within 7e-5 on features and 2e-6 on probabilities, with identical predicted classes.
