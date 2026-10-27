"""Run the MediaPipe HandLandmarker over the personal test set and cache the landmarks.

python scripts/extract_personal_landmarks.py [--data data/personal] [--out data/processed/personal_landmarks.npz]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from hgr.labels import PERSONAL_TO_HAGRID
from hgr.landmarker import HandLandmarker

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "personal")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed" / "personal_landmarks.npz")
    args = parser.parse_args()

    landmarks, aspects, labels, handedness, paths, missed = [], [], [], [], [], []
    with HandLandmarker(max_hands=1) as landmarker:
        for folder, label in PERSONAL_TO_HAGRID.items():
            for path in sorted((args.data / folder).glob("*.jpg")):
                bgr = cv2.imread(str(path))
                hands = landmarker.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                rel = path.relative_to(args.data).as_posix()
                if not hands:
                    missed.append(rel)
                    continue
                pts, side, _ = hands[0]
                landmarks.append(pts)
                aspects.append(bgr.shape[1] / bgr.shape[0])
                labels.append(label)
                handedness.append(side)
                paths.append(rel)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        landmarks=np.stack(landmarks),
        aspect=np.array(aspects, dtype=np.float32),
        label=np.array(labels),
        handedness=np.array(handedness),
        path=np.array(paths),
    )
    print(f"Saved {len(landmarks)} hands to {args.out}")
    if missed:
        print(f"No hand detected in {len(missed)} image(s): {', '.join(missed)}")


if __name__ == "__main__":
    main()
