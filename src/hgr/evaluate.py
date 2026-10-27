"""Evaluate a trained classifier on the HaGRIDv2 test split and the personal test set.

    python -m hgr.evaluate [--model models/gesture_mlp.pt] [--out reports/metrics.json]

HaGRID test (unseen subjects) is reported two ways:
- core: the six gestures + HaGRID's no_gesture class;
- open-set: additionally every other HaGRID gesture, which must be rejected as no_gesture.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from hgr.augment import augmented_features
from hgr.features import featurize
from hgr.hagrid import GENDERS, RACES, load_cache
from hgr.labels import TARGET_CLASSES
from hgr.metrics import classification_metrics, expected_calibration_error
from hgr.model import GestureClassifier
from hgr.train import select, selection_mask

REPO_ROOT = Path(__file__).resolve().parents[2]
GESTURES = [i for i, c in enumerate(TARGET_CLASSES) if c != "no_gesture"]


def hagrid_test(classifier, arrays, negatives, seed=2):
    lm, y = select(arrays, 2, negatives)
    probs = classifier.predict_proba(augmented_features(lm, np.random.default_rng(seed), augment=False))
    metrics = classification_metrics(y, probs.argmax(1), TARGET_CLASSES, GESTURES)
    metrics["ece"] = expected_calibration_error(probs, y)
    return metrics, probs.argmax(1), y


def demographic_slices(arrays, pred, y):
    """Gesture recall by auto-annotated gender and race of the subject (core test set)."""
    keep = selection_mask(arrays, 2, "no_gesture")
    slices = {}
    for field, names in (("gender", GENDERS), ("race", RACES)):
        values = arrays[field][keep]
        for i, name in enumerate(names):
            mask = (values == i) & np.isin(y, GESTURES)
            if mask.sum():
                slices[f"{field}={name}"] = {
                    "n": int(mask.sum()),
                    "gesture_recall": float((pred[mask] == y[mask]).mean()),
                }
    return slices


def personal(classifier, path):
    data = np.load(path)
    y = np.array([TARGET_CLASSES.index(c) for c in data["label"]])
    probs = classifier.predict_proba(featurize(data["landmarks"], data["aspect"]))
    pred = probs.argmax(1)
    metrics = classification_metrics(y, pred, TARGET_CLASSES, gesture_indices=GESTURES)
    metrics["errors"] = [
        {"path": str(p), "true": TARGET_CLASSES[t], "pred": TARGET_CLASSES[q], "confidence": float(pr.max())}
        for p, t, q, pr in zip(data["path"], y, pred, probs, strict=True)
        if t != q
    ]
    return metrics


def summary_line(name, m):
    return (f"{name:28s} n={m['n']:>7,}  accuracy {m['accuracy']:.1%}  macro precision {m['macro_precision']:.1%}"
            f"  macro recall {m['macro_recall']:.1%}  macro F1 {m['macro_f1']:.1%}")  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "gesture_mlp.pt")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "processed" / "hagrid_landmarks.npz")
    parser.add_argument("--personal", type=Path, default=REPO_ROOT / "data" / "processed" / "personal_landmarks.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "metrics.json")
    args = parser.parse_args()

    classifier = GestureClassifier.load(args.model)
    arrays = load_cache(args.cache)
    core, core_pred, core_y = hagrid_test(classifier, arrays, "no_gesture")
    open_set, _, _ = hagrid_test(classifier, arrays, "all")
    report = {
        "model": str(args.model),
        "temperature": classifier.temperature,
        "reject_bias": classifier.reject_bias,
        "hagrid_test_core": core,
        "hagrid_test_open_set": open_set,
        "hagrid_test_core_slices": demographic_slices(arrays, core_pred, core_y),
        "personal": personal(classifier, args.personal),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    print(summary_line("HaGRID test (core)", core))
    print(summary_line("HaGRID test (open-set)", open_set))
    print(summary_line("Personal photos", report["personal"]))
    print(f"ECE core {core['ece']:.3f}  open-set {open_set['ece']:.3f}")
    for name, s in report["hagrid_test_core_slices"].items():
        print(f"  {name:14s} n={s['n']:>6,}  gesture recall {s['gesture_recall']:.1%}")
    for e in report["personal"]["errors"]:
        print(f"  personal error: {e['path']}  {e['true']} -> {e['pred']} ({e['confidence']:.0%})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
