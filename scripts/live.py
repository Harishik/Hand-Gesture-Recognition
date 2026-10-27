"""Live webcam gesture recognition. Press Q to quit.

python scripts/live.py [--model models/gesture_mlp.pt] [--camera 0]
"""

import argparse
import time
from pathlib import Path

import cv2

from hgr.model import GestureClassifier
from hgr.pipeline import GesturePipeline, draw

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "gesture_mlp.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=0.6)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    start = time.perf_counter()
    fps = 0.0
    with GesturePipeline(GestureClassifier.load(args.model), smoothing=args.smoothing) as pipe:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            preds = pipe.process(frame, (t0 - start) * 1000)
            fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t0, 1e-6)
            display = draw(cv2.flip(frame, 1), [_mirror(p) for p in preds])
            cv2.putText(display, f"{fps:.0f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Hand Gesture Recognition", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


def _mirror(pred):
    """Landmarks for a horizontally flipped (selfie-view) display."""
    pts = pred.landmarks.copy()
    pts[:, 0] = 1 - pts[:, 0]
    pred.landmarks = pts
    return pred


if __name__ == "__main__":
    main()
