"""Record labelled webcam clips for real-time evaluation: one clip per gesture, guided on screen.

    python scripts/record_clips.py --session alice_daylight [--seconds 8] [--camera 0]

Press SPACE to start each gesture (a 2-second countdown runs first and is not saved), Q to quit.
For `no_gesture`, keep hands in view but relaxed, moving naturally, or doing gestures outside the six classes.
Clips go to data/clips/<session>/<gesture>.mp4. Record several sessions (people, rooms, lighting) for a fair test.
"""

import argparse
import time
from pathlib import Path

import cv2

from hgr.labels import TARGET_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = {
    "palm": "Open palm facing the camera",
    "fist": "Closed fist",
    "peace": "Peace / V sign",
    "ok": "OK sign (thumb and index touching)",
    "like": "Thumbs up",
    "dislike": "Thumbs down",
    "no_gesture": "Hands visible but relaxed / moving / other gestures",
}


def overlay(frame, lines, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="name for this recording session, e.g. alice_daylight")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "clips")
    args = parser.parse_args()

    out_dir = args.out / args.session
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for gesture in TARGET_CLASSES:
        while True:  # wait for SPACE
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Camera read failed")
            overlay(frame, [f"Next: {gesture}", INSTRUCTIONS[gesture], "SPACE = start, Q = quit"])
            cv2.imshow("record", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key == ord(" "):
                break

        countdown_end = time.time() + 2.0
        while time.time() < countdown_end:
            ok, frame = cap.read()
            overlay(frame, [f"{gesture} in {countdown_end - time.time():.1f}s"], (0, 200, 255))
            cv2.imshow("record", frame)
            cv2.waitKey(1)

        h, w = frame.shape[:2]
        path = out_dir / f"{gesture}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        end = time.time() + args.seconds
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            preview = overlay(frame.copy(), [f"REC {gesture}  {end - time.time():.1f}s"], (0, 0, 255))
            cv2.imshow("record", preview)
            cv2.waitKey(1)
        writer.release()
        print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
