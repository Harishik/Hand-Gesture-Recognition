# Auditing HaGRIDv2 for Landmark-Based Gesture Recognition: Leakage, Hidden Aspect Ratios, Open-Set Rejection, and Cross-Dataset Transfer

*Working draft, to be submitted as a short paper or workshop paper.*

## Abstract

- **Setting:** landmark-based hand gesture recognition is the standard design for real-time, on-device systems. MediaPipe-style 21-point hand landmarks feed a small classifier. HaGRIDv2 (1.09M images, 33 gestures) is its largest public training source.
- **Audit findings:** we audit HaGRIDv2 for this use and report three issues that change how models trained on it should be built and evaluated:
  - The official train/test split is not subject-disjoint.
  - Landmarks are normalised by an image size the annotations do not record, across mixed aspect ratios that we measure: 3:4 portrait in 66.7% of images, 4:3 in 15.2%, 9:16 in 9.5%, 16:9 in 6.8% and 1:1 in 1.8%.
  - The "inverted" classes are back-of-hand views, not rotations.
- **Models:** we then train compact MLPs on 67 geometric landmark features, plus a pair model for two-handed gestures.
  - The one-hand model reaches 98.8% accuracy and 98.9% macro precision over 26 gestures + rejection on 153,755 test hands from unseen subjects.
  - The pair model reaches 99.3% accuracy and 98.0% macro precision on 7 two-handed gestures.
- **Open-set result:** the choice of negatives dominates real-world precision. Training rejection only on HaGRID's `no_gesture` class gives 32.4% gesture precision when other hand shapes appear, against 94.2% with all other gestures as negatives, with no loss on the closed-set test.
- **Cross-dataset transfer:** on the independent HANDS dataset (Kinect v2, 5 subjects), 81.0% of detected hands are classified correctly (72.9% end to end, 700 samples), with near-perfect transfer on open palm and fist.
- **Deployment:** the system runs in a web browser at about 25–30 FPS, with a JavaScript inference path verified against PyTorch to 2×10⁻⁶ in probability.

## 1. Introduction

- **Why this design:** touchless interfaces (smart TVs, kiosks, AR/VR, automotive) need gesture recognition that is accurate, fast, private and robust to people and places it has never seen. A two-stage design meets these constraints better than end-to-end image classifiers:
  - a pretrained hand-landmark model, such as MediaPipe Hands;
  - followed by a light classifier over landmarks.
  The classifier is tiny, invariant to appearance and background by construction, and easy to run on-device.
- **Why the data needs scrutiny:** the quality of such systems is bounded by the landmark training data. HaGRIDv2 [1] provides MediaPipe landmarks for over a million hands, which makes it the natural source. Its annotations, however, were designed for image-level detection and classification, and some properties that matter for landmark models are undocumented.

**Contributions.**
1. **Dataset audit:** a subject leak in the official split, an undocumented aspect-ratio mixture (measured by image-header probing), and the semantics of "inverted" classes, each with a fix.
2. **Open-set rejection:** evidence that negative-class design, not model capacity, determines precision under open-set conditions, by a factor of three.
3. **Models:** a complete 33-gesture model pair (one-hand MLP plus two-hand pair MLP) with calibrated confidence and per-demographic results.
4. **Cross-dataset evaluation** on an independent dataset (HANDS, Kinect v2): 81.0% accuracy on detected hands without any fine-tuning.
5. **Reproducible deployment:** an open-source, on-device browser implementation with verified Python/JavaScript parity.

## 2. Related work

**Datasets.**
- **HaGRID / HaGRIDv2 [1, 2]:** the largest static gesture image datasets.
- **Jester [3] and IPN Hand [4]:** dynamic gestures.
- **NVGesture [5]:** multimodal in-car gestures.
- **HANDS [6]:** static gestures for human–robot interaction, recorded with a Kinect v2.
- **3D hand tracking datasets from industry:** Meta's UmeTrack and HOT3D [7], and Google's FSboard fingerspelling corpus [8]. These provide landmarks or poses but no gesture labels compatible with interface gestures.

**Landmark-based recognition.**
- MediaPipe's Gesture Recognizer pairs hand landmarks with a small classifier head.
- Recent work explores transformers over joint coordinates [9, 10] and graph networks over the hand skeleton [11], and some systems pair landmarks with vision-language models for free-form gestures [12].
- Few-shot gesture customisation from a single demonstration has been shown with meta-learned landmark embeddings [13].
- Training-free geometric detectors handle some dynamic gestures exactly [14].

**3D hand reconstruction.** End-to-end 3D hand reconstruction (WiLoR [15], HaMeR) offers an alternative landmark source that is more robust to close-ups, at higher cost.

## 3. Auditing HaGRIDv2 for landmark models

**Setup.** We parse the HaGRIDv2 annotations (boxes, labels, user IDs, MediaPipe landmarks) into 1,308,352 hands with landmarks. 195,781 annotated hands (13%) have no landmarks and are excluded. The supplied split has 983,991 / 123,230 / 201,131 hands in train / val / test, across 55,892 / 4,159 / 4,107 subjects.

**3.1 Subject leakage.**
- **Finding:** two subjects appear in both train and test, with 1 train image each and 4 and 8 test images.
- **Fix:** we move each subject wholly into the split holding most of their samples, which moves 2 hands.
- **Impact:** small on HaGRID, but it violates the subject-independence that the split is meant to guarantee. We release both splits.

**3.2 Hidden aspect ratios.**
- **Why it matters:** landmarks are given as fractions of image width and height, so recovering hand geometry requires the image aspect ratio, which the annotations omit.
- **Symptom:** for upright open hands (`palm`, `stop`, `four`), the palm width/length ratio computed in raw coordinates is clearly multi-modal (Fig. 1).
- **Measurement:** we range-read only the JPEG headers of 400 random images from a public 384p HaGRID sample and match 384 of them to annotations by image ID. The measured mixture is 3:4 = 66.7%, 4:3 = 15.2%, 9:16 = 9.5%, 16:9 = 6.8%, 1:1 = 1.8%.
- **Validation:** applying each image's true aspect collapses the palm ratio of portrait hands (raw median 0.95) and landscape hands (0.51) onto 0.70 and 0.71. That matches the 0.70 from MediaPipe's metric 3D world landmarks on independent photos.
- **Our approach:** a single assumed aspect would distort about a third of the data, so we sample the aspect from the measured prior during training and use the true camera aspect at inference.

**3.3 "Inverted" classes.**
- **Finding:** HaGRID's `peace_inverted`, `stop_inverted` and `two_up_inverted` denote the back of the hand facing the camera, not upside-down hands.
- **How we found it:** a model trained with `peace_inverted` as a negative rejected back-of-hand peace signs in an independent personal set with 95–100% confidence, and a nearest-neighbour analysis placed those hands among `peace_inverted` examples (22–25 of 25 neighbours).
- **Implications:** treating these as separate classes, as we do in the 26-gesture model, is faithful to HaGRID. Interfaces that only care about the hand shape should merge them; we report "lenient" scores that do.

## 4. Method

**Features (67-D).**
- **Normalisation:** landmarks are converted to isotropic units with the aspect ratio, translated to the wrist, and scaled by palm length (wrist to middle-finger MCP).
- **Contents:** the 42 normalised coordinates, 15 joint bend angles and 10 fingertip pairwise distances.
- **Rotation:** deliberately kept, because thumbs-up and thumbs-down differ only in orientation.

**Pair features (221-D).** For images with two annotated hands:
- both hands' points in a shared frame, centred between the wrists and scaled by mean palm length;
- each hand's own 67 features;
- the wrist-to-wrist vector and distance.

The features are invariant to hand order.

**Models.**
- **Architecture:** a 3×256 MLP with BatchNorm, GELU and dropout 0.1.
- **Training:** class-balanced epochs of 30,000 samples per class, freshly augmented each epoch:
  - aspect drawn from the prior;
  - mirror;
  - ±20° rotation;
  - ±8% anisotropic stretch;
  - landmark jitter.
- **Optimiser:** AdamW with a one-cycle schedule (2×10⁻³), label smoothing 0.05, 30 epochs.
- **Calibration:** temperature scaling on validation, then a rejection-logit bias tuned on validation to maximise class-balanced macro F1. Test data is never used for any choice.

**Real-time system.**
- **Smoothing:** an exponential moving average of class probabilities per tracked hand.
- **Events:** dwell-to-confirm, where a gesture must be held for 450 ms to fire once.
- **Motion:** swipes from palm-centre velocity normalised by palm size.
- **Pinch:** thumb–index distance with hysteresis, driving a One Euro-filtered cursor.
- **Deployment:** the classifiers are exported with BatchNorm folded and run in dependency-free JavaScript next to MediaPipe's WebAssembly/WebGL landmarker.

## 5. Experiments

**5.1 In-domain results** (HaGRIDv2 test, unseen subjects).

| Model | Test set | n | Acc. | Macro P | Macro R | Macro F1 | ECE |
|---|---|---|---|---|---|---|---|
| One-hand, 26 gestures | core (26 + no_gesture) | 153,755 | 98.77 | 98.91 | 98.86 | 98.88 | 0.014 |
| | open-set (+ two-hand halves) | 201,133 | 94.66 | 95.66 | 98.85 | 96.75 | 0.021 |
| One-hand, 6 gestures | core | 60,488 | 99.37 | 99.21 | 99.61 | 99.41 | 0.002 |
| | open-set (+ 26 other gestures) | 201,133 | 98.96 | 94.19 | 99.62 | 96.81 | 0.002 |
| Two-hand, 7 gestures | pairs | 44,195 | 99.28 | 98.01 | 98.89 | 98.45 | — |

- **Hardest one-hand classes** (F1): `one` 0.964, `mute` 0.965 and `point` 0.966. All are index-finger poses that differ mainly in orientation.
- **Hardest two-hand class:** `timeout` (P 0.941 / R 0.966).

**5.2 Negative-class design (open-set).** These are 6-gesture models evaluated on the full test set, where the 26 other gestures must be rejected.

| Negatives during training | Core acc. | Open-set macro P | Open-set acc. |
|---|---|---|---|
| `no_gesture` only | 99.5 | **32.4** | 61.5 |
| all other HaGRID gestures | 99.4 | **94.1** | 99.0 |

- **Closed-set:** both models look equally good.
- **Open-set:** with only `no_gesture` as negatives, two out of three "gesture" predictions on other hand shapes are wrong.

**5.3 Fairness.** Gesture recall on the core test set, by the subjects' auto-annotated attributes:

| Group | female | male | White | Black | Asian | Indian |
|---|---|---|---|---|---|---|
| n | 70,282 | 57,592 | 111,955 | 6,197 | 5,415 | 2,476 |
| Recall | 98.90 | 98.81 | 98.97 | 98.11 | 98.95 | 98.59 |

**5.4 Cross-dataset transfer (HANDS [6]).** Both models were applied unchanged to HANDS Subject 1 (every third frame; 700 one-hand and 100 two-hand annotated samples). MediaPipe detected 90% of annotated single hands; of those, 81.0% were classified correctly (72.9% end to end; precision of named predictions 82.3%; 1.6% rejected as no gesture). Transfer is near-perfect for Five→palm (99.0%) and Punch→fist (98.7%) and strong for Three (93.9%) and Four (92.9%), but weaker for Two→peace (75.3%), One (68.3%) and Span→call (30.5%), whose HANDS executions sit between HaGRID classes. For two-hand gestures the bottleneck is detection, not classification: both hands were found in only 6% of frames, all of which were classified correctly. Remaining subjects (2–5) are left to future work owing to storage limits.
- **Protocol:** the models are trained on HaGRIDv2 only. HANDS frames are processed with the same MediaPipe landmarker, and labels are mapped to HaGRID classes where the gestures coincide.
- **What we report:** accuracy and macro precision on the mapped classes, rejection behaviour on HANDS gestures without a HaGRID equivalent, and the two-hand model on `XSign` and `TimeOut`.

**5.5 Latency and parity.**
- **Classifier:** about 1 ms on CPU.
- **End to end:** the Python pipeline runs at about 21 FPS on a laptop CPU, and the browser at about 25–30 FPS.
- **A pitfall:** running PyTorch with its default thread pool in the same process slowed MediaPipe's landmarker 2.5× (38 → 95 ms). Limiting PyTorch to one thread removes the slowdown.
- **Parity:** JavaScript features match Python to within 7×10⁻⁵, and probabilities to within 2×10⁻⁶, with identical predicted classes.

## 6. Limitations

- **Labels are MediaPipe's own output:** HaGRID's landmarks were produced by MediaPipe, and hands it failed on are excluded, so in-domain numbers measure the classifier on hands the landmarker already handles. The cross-dataset study is meant to address this.
- **2D only:** HaGRID provides 2D landmarks, so depth features are not used for training.
- **Static only:** dynamic gestures beyond swipes are not learned.
- **Auto-annotated demographics:** the attributes are auto-annotated by face models and are only approximate.

## 7. Conclusion

A 67-feature landmark MLP trained on subject-disjoint HaGRIDv2 reaches 98.8% accuracy on unseen people, stays within one point across demographic slices, runs in about 1 ms, and transfers to an independent Kinect dataset at 81% on detected hands. The cross-dataset gaps point to gesture-definition mismatch and two-hand detection as the main open problems, not classifier capacity.

## References

1. A. Nuzhdin et al. HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition. arXiv:2412.01508, 2024.
2. A. Kapitanov et al. HaGRID: HAnd Gesture Recognition Image Dataset. WACV 2024. arXiv:2206.08219.
3. J. Materzynska et al. The Jester Dataset: A Large-Scale Video Dataset of Human Gestures. ICCVW 2019.
4. G. Benitez-Garcia et al. IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition. ICPR 2020. arXiv:2005.02134.
5. P. Molchanov et al. Online Detection and Classification of Dynamic Hand Gestures with Recurrent 3D CNNs. CVPR 2016 (NVGesture).
6. C. Nuzzi, S. Pasinetti, R. Pagani, G. Coffetti, G. Sansoni. HANDS: an RGB-D dataset of static hand-gestures for human-robot interaction. Data in Brief 35, 106791, 2021. doi:10.1016/j.dib.2021.106791.
7. P. Banerjee et al. Introducing HOT3D: An Egocentric Dataset for 3D Hand and Object Tracking. arXiv:2406.09598, 2024.
8. M. Georg et al. FSboard: Over 3 million characters of ASL fingerspelling collected via smartphones. arXiv:2407.15806, 2024.
9. TSLFormer: A Lightweight Transformer Model for Turkish Sign Language Recognition Using Skeletal Landmarks. arXiv:2505.07890, 2025.
10. Real-Time Hand Gesture Recognition for OpenXR Using Transformer-Based Machine Learning. arXiv:2609.25466, 2026.
11. Improving Handshape Representations for Sign Language Processing: A Graph Neural Network Approach. arXiv:2509.18309, 2025.
12. Gestura: A LVLM-Powered System Bridging Motion and Semantics for Real-Time Free-Form Gesture Understanding. arXiv:2510.21814, 2025.
13. Vision-Based Hand Gesture Customization from a Single Demonstration. arXiv:2402.08420, 2024.
14. Frame-Synchronous Hand Gesture Detection by Projected Winding Order. arXiv:2609.13269, 2026.
15. R. A. Potamias et al. WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild. arXiv:2409.12259, 2024.

*Author lists marked "et al." and titles of refs 9–14 should be verified against the arXiv pages before submission.*
