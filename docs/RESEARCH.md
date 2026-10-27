# Research notes: what the current state of the art does, and what this project adopts

A survey of recent papers, products and open-source projects (October 2026). For each idea: what it is, where it comes from, and whether this project uses it.

## Recognition

| Idea | Source | Used here |
|---|---|---|
| Two-stage pipeline: hand landmarks first, then a small classifier on landmarks, instead of classifying raw pixels | MediaPipe Gesture Recognizer; [HaGRIDv2](https://arxiv.org/abs/2412.01508) baselines; [hand-gesture-classification-hagrid](https://github.com/sokaryy/hand-gesture-classification-hagrid) | ✅ MediaPipe HandLandmarker plus MLPs on 67 landmark features. Robust to lighting, background and skin tone, and runs in under 1 ms. |
| Large, diverse training data, split by subject | [HaGRIDv2](https://github.com/hukenovs/hagrid): 1M+ images, 33 gestures | ✅ 1.31M hands, with a subject-disjoint split (leak fixed, see [DATA.md](DATA.md)) |
| Open-set rejection: training on many "other" gestures so unknown hand shapes don't trigger | HaGRID's `no_gesture` class; open-set recognition literature | ✅ Ablation in [MODEL.md](MODEL.md): without it, gesture precision on unknown hand shapes drops to 32% |
| Two-hand gestures recognised from both hands jointly | HaGRIDv2's two-handed classes (`hand_heart`, `timeout`, …) | ✅ A dedicated pair model (99.3% test accuracy) |
| Dynamic (motion) gestures from landmark trajectories | HaGRIDv2's [dynamic_gestures](https://github.com/ai-forever/dynamic_gestures) (dynamic gestures from static detections plus motion); [kapi0okapi/hand-gesture-recognition](https://github.com/kapi0okapi/hand-gesture-recognition) (causal TCN) | ✅ Swipes from palm-centre velocity, normalised by palm size. A learned temporal model (TCN/transformer) is future work. |
| Transformers over joint coordinates | [Real-Time Hand Gesture Recognition for OpenXR Using Transformers](https://arxiv.org/pdf/2609.25466); GestFormer (CVPRW 2024) | ⏳ Planned as an ablation against the MLP (Phase 4) |
| Calibrated confidence | Temperature scaling (Guo et al., 2017) | ✅ ECE 0.002 on test |
| Few-shot personalisation | Prototype / nearest-neighbour learning in embedding space | ✅ "Teach" tab: a custom gesture from 3 s of examples, matched in the MLP's learned embedding |

## Interaction design (XR and touchless UI)

| Idea | Source | Used here |
|---|---|---|
| Pinch as the primary "click", plus a cursor ray or pointer | [Meta Horizon OS input](https://developers.meta.com/horizon/essentials/horizon-os-input/); [Apple Vision Pro](https://uploadvr.com/apple-vision-pro-gesture-controls) | ✅ Air cursor: index fingertip moves it, pinch clicks or drags any control |
| Immediate visible feedback when an input is recognised | [Google: gaze + pinch design principles](https://research.google/pubs/design-principles-and-challenges-for-gaze-pinch-interaction-in-xr/); visionOS guidance | ✅ Live label on each hand, hold-progress ring, toast and event log on every action |
| Dwell / hold-to-confirm, to avoid accidental triggers (the "Midas touch" problem) | Long-standing HCI practice for touchless input | ✅ A gesture must be held for 450 ms (adjustable) and fires once per hold |
| Low-latency jitter filtering for pointing | One Euro filter (Casiez et al., CHI 2012), standard in XR SDKs | ✅ The cursor uses a One Euro filter |
| Avoid sustained arm raising ("gorilla arm") | visionOS and Meta comfort guidance | ✅ The central 70% of the camera view maps to the whole screen, so small movements reach every corner |
| Input-agnostic UI: everything also works with mouse and touch | Meta Horizon OS guidance | ✅ Every control works with mouse, touch or gestures |
| On-device processing for privacy | Industry norm for camera input | ✅ Everything runs in the browser tab; no frames are uploaded |

## Existing open-source projects, and the gap

Most open-source "gesture control" projects are desktop Python scripts that move the OS cursor with PyAutoGUI, for example [AirControl](https://awesome.ecosyste.ms/projects/github.com%2Fnexustech101%2Faircontrol) and [AI Virtual Mouse](https://github.com/singharyans754/AI-Virtual-Mouse). The one browser-based attempt found, [touchless-ui](https://github.com/yujie-tao/touchless-ui), only plays and stops YouTube videos.

None combines:
- a model trained and evaluated on a large dataset;
- two-hand and motion gestures;
- calibrated rejection;
- a polished browser front-end that runs fully on-device.

That combination is this project's niche.
