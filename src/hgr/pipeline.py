"""Real-time gesture pipeline: MediaPipe hand landmarks -> features -> calibrated classifier -> temporal smoothing."""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from hgr.features import featurize
from hgr.landmarker import HandLandmarker

# Pairs of landmark indices forming the hand skeleton, for drawing.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip


@dataclass
class HandPrediction:
    label: str
    confidence: float
    probs: np.ndarray  # smoothed class probabilities
    raw_label: str  # this frame's prediction before smoothing
    landmarks: np.ndarray  # (21, 2) image-normalised
    handedness: str


class GesturePipeline:
    """Per-frame inference with an exponential moving average of class probabilities per hand.

    Hands are matched across frames by MediaPipe's handedness label, which is stable while a hand stays in view;
    a hand's history is dropped when it disappears.
    """

    def __init__(self, classifier, smoothing=0.6, max_hands=2, min_detection_confidence=0.5):
        # The MLP needs one thread. PyTorch's default pool competes with MediaPipe's for cores and measured 2.5x
        # slower hand landmarking (38 -> 95 ms per frame on a 12-thread laptop CPU).
        torch.set_num_threads(1)
        self.classifier = classifier
        self.smoothing = smoothing  # weight on the previous average; 0 disables smoothing
        self.landmarker = HandLandmarker(max_hands=max_hands, min_confidence=min_detection_confidence, video=True)
        self._history = {}
        self._last_ts = -1
        self.timings = {"landmarks_ms": [], "classify_ms": []}

    def process(self, frame_bgr, timestamp_ms):
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)  # MediaPipe requires strictly increasing stamps
        self._last_ts = timestamp_ms
        h, w = frame_bgr.shape[:2]

        t0 = time.perf_counter()
        hands = self.landmarker.detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), timestamp_ms)
        t1 = time.perf_counter()
        results = []
        if hands:
            probs = self.classifier.predict_proba(featurize(np.stack([pts for pts, _, _ in hands]), w / h))
            seen = set()
            for (pts, side, _), p in zip(hands, probs, strict=True):
                key = side if side not in seen else f"{side}#2"
                seen.add(key)
                prev = self._history.get(key)
                smoothed = p if prev is None else self.smoothing * prev + (1 - self.smoothing) * p
                self._history[key] = smoothed
                top = int(smoothed.argmax())
                results.append(
                    HandPrediction(
                        label=self.classifier.classes[top],
                        confidence=float(smoothed[top]),
                        probs=smoothed,
                        raw_label=self.classifier.classes[int(p.argmax())],
                        landmarks=pts,
                        handedness=side,
                    )
                )
            self._history = {k: v for k, v in self._history.items() if k in seen}
        else:
            self._history = {}
        t2 = time.perf_counter()
        self.timings["landmarks_ms"].append((t1 - t0) * 1000)
        self.timings["classify_ms"].append((t2 - t1) * 1000)
        return results

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw(frame, predictions, color=(0, 220, 0)):
    h, w = frame.shape[:2]
    for pred in predictions:
        pts = (pred.landmarks * [w, h]).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, color, -1)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        text = f"{pred.label} {pred.confidence:.0%}"
        cv2.putText(frame, text, (x, max(25, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return frame
