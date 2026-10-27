"""Train the two-handed gesture classifier on pairs of HaGRIDv2 hands from the same image.

python -m hgr.train_pairs [--out models/pair_mlp.pt]
"""

import argparse
from pathlib import Path

import numpy as np

from hgr.hagrid import load_cache
from hgr.metrics import classification_metrics
from hgr.pairs import PAIR_CLASSES, PAIR_FEATURE_DIM, augmented_pair_features, build_pairs
from hgr.train import REPO_ROOT, add_training_args, fit, save


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_training_args(parser)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "pair_mlp.pt")
    parser.set_defaults(per_class=20000)
    args = parser.parse_args()

    arrays = load_cache(args.cache)
    lm_train, y_train = build_pairs(arrays, 0)
    lm_val, y_val = build_pairs(arrays, 1)
    lm_test, y_test = build_pairs(arrays, 2)
    print(f"pairs train {len(y_train):,}  val {len(y_val):,}  test {len(y_test):,}", flush=True)
    print("train counts", dict(zip(PAIR_CLASSES, np.bincount(y_train).tolist(), strict=True)), flush=True)

    x_val = augmented_pair_features(lm_val, np.random.default_rng(1), augment=False)

    def make_features(idx, rng):
        return augmented_pair_features(lm_train[idx], rng, max_rotation_deg=args.max_rotation)

    clf, report = fit(make_features, y_train, x_val, y_val, PAIR_CLASSES, args, in_dim=PAIR_FEATURE_DIM, kind="pair")
    x_test = augmented_pair_features(lm_test, np.random.default_rng(2), augment=False)
    gestures = [i for i, c in enumerate(PAIR_CLASSES) if c != "no_gesture"]
    report["test"] = classification_metrics(y_test, clf.predict(x_test), PAIR_CLASSES, gestures)
    t = report["test"]
    print(
        f"test accuracy {t['accuracy']:.4f}  macro precision {t['macro_precision']:.4f}  recall {t['macro_recall']:.4f}"
    )
    save(clf, report, args.out)


if __name__ == "__main__":
    main()
