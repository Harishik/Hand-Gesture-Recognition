"""Thin wrapper around the MediaPipe Tasks HandLandmarker (the current API, not legacy `mediapipe.solutions`)."""

import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"


def ensure_model(path=DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandLandmarker:
    """Detects hands in RGB images and returns their 21 landmarks in image-normalised coordinates."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, max_hands=2, min_confidence=0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb):
        """rgb: HxWx3 uint8. Returns a list of (landmarks (21, 2) float32, handedness str, score float)."""
        import mediapipe as mp

        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        hands = []
        for lms, handedness in zip(result.hand_landmarks, result.handedness, strict=True):
            pts = np.array([(p.x, p.y) for p in lms], dtype=np.float32)
            hands.append((pts, handedness[0].category_name, handedness[0].score))
        return hands

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
