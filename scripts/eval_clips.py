"""Real-time evaluation on recorded clips: frame-level accuracy / precision of the full pipeline, plus latency.

    python scripts/eval_clips.py [--clips data/clips] [--model models/gesture_mlp.pt] [--warmup 0.5]

Every clip is named after its label (data/clips/<session>/<label>.mp4). Each frame is scored on the most confident
hand's smoothed prediction; a frame with no detected hand counts as `no_gesture`, exactly as the live app behaves.
The first `--warmup` seconds of each clip are skipped (the hand is still moving into the pose).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hgr.labels import TARGET_CLASSES
from hgr.metrics import classification_metrics
from hgr.model import GestureClassifier
from hgr.pipeline import GesturePipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
REJECT = TARGET_CLASSES.index("no_gesture")
GESTURES = [i for i in range(len(TARGET_CLASSES)) if i != REJECT]  # macro averages: the six gestures


def run_clip(path, classifier, smoothing, warmup_s):
    """Returns (smoothed predictions, raw predictions, hand-detected flags, pipeline timings) for scored frames."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    smoothed, raw, detected = [], [], []
    with GesturePipeline(classifier, smoothing=smoothing) as pipe:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            preds = pipe.process(frame, t * 1000)
            frame_idx += 1
            if t < warmup_s:
                continue
            if preds:
                best = max(preds, key=lambda p: p.confidence if p.label != "no_gesture" else 0.0)
                smoothed.append(TARGET_CLASSES.index(best.label))
                raw.append(TARGET_CLASSES.index(best.raw_label))
            else:
                smoothed.append(REJECT)
                raw.append(REJECT)
            detected.append(bool(preds))
        timings = pipe.timings
    cap.release()
    return smoothed, raw, detected, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, default=REPO_ROOT / "data" / "clips")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "gesture_mlp.pt")
    parser.add_argument("--smoothing", type=float, default=0.6)
    parser.add_argument("--warmup", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "realtime_metrics.json")
    args = parser.parse_args()

    classifier = GestureClassifier.load(args.model)
    clips = sorted(p for p in args.clips.glob("*/*.mp4") if p.stem in TARGET_CLASSES)
    if not clips:
        raise SystemExit(f"No clips found under {args.clips}; record some with scripts/record_clips.py")

    y_true, y_smooth, y_raw, detected = [], [], [], []
    land_ms, cls_ms, per_session = [], [], {}
    for path in clips:
        smoothed, raw, det, timings = run_clip(path, classifier, args.smoothing, args.warmup)
        label = TARGET_CLASSES.index(path.stem)
        y_true += [label] * len(smoothed)
        y_smooth += smoothed
        y_raw += raw
        detected += det
        land_ms += timings["landmarks_ms"]
        cls_ms += timings["classify_ms"]
        session = per_session.setdefault(path.parent.name, {"true": [], "pred": []})
        session["true"] += [label] * len(smoothed)
        session["pred"] += smoothed
        acc = np.mean(np.array(smoothed) == label) if smoothed else float("nan")
        print(
            f"{path.parent.name}/{path.stem:11s} frames {len(smoothed):4d}"
            f"  hand found {np.mean(det):.0%}  acc {acc:.1%}"
        )

    y_true, y_smooth, y_raw = map(np.array, (y_true, y_smooth, y_raw))
    total_ms = np.array(land_ms) + np.array(cls_ms)
    report = {
        "model": str(args.model),
        "smoothing": args.smoothing,
        "warmup_s": args.warmup,
        "clips": len(clips),
        "frames": int(len(y_true)),
        "hand_detection_rate": float(np.mean(detected)),
        "smoothed": classification_metrics(y_true, y_smooth, TARGET_CLASSES, GESTURES),
        "raw": classification_metrics(y_true, y_raw, TARGET_CLASSES, GESTURES),
        "per_session_accuracy": {
            name: float(np.mean(np.array(s["pred"]) == np.array(s["true"]))) for name, s in per_session.items()
        },
        "latency_ms": {
            "landmarks_mean": float(np.mean(land_ms)),
            "classify_mean": float(np.mean(cls_ms)),
            "total_mean": float(total_ms.mean()),
            "total_p95": float(np.percentile(total_ms, 95)),
            "throughput_fps": float(1000 / total_ms.mean()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    for name in ("raw", "smoothed"):
        m = report[name]
        print(f"\n{name:8s} accuracy {m['accuracy']:.1%}  macro precision {m['macro_precision']:.1%}"
              f"  macro recall {m['macro_recall']:.1%}  macro F1 {m['macro_f1']:.1%}")  # fmt: skip
    lat = report["latency_ms"]
    print(
        f"\nlatency: {lat['total_mean']:.1f} ms mean, {lat['total_p95']:.1f} ms p95 -> {lat['throughput_fps']:.0f} FPS"
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
