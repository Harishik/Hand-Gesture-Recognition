"""Dataset figures for docs/: class balance per split and the palm-shape histogram used to diagnose image aspect.

python scripts/plot_dataset.py [--cache data/processed/hagrid_landmarks.npz] [--out docs/figures]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hgr.hagrid import SPLITS, load_cache  # noqa: E402
from hgr.labels import HAGRID_CLASSES, TARGET_CLASSES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OPEN_HAND_CLASSES = ("palm", "stop", "four")


def palm_ratio(landmarks, aspect=1.0, min_vertical=0.95):
    """Palm width / palm length for hands whose palm axis is near-vertical (x rescaled by `aspect`)."""
    pts = landmarks.astype(np.float64).copy()
    pts[..., 0] *= aspect
    width = np.linalg.norm(pts[:, 5] - pts[:, 17], axis=-1)
    axis = pts[:, 9] - pts[:, 0]
    length = np.linalg.norm(axis, axis=-1)
    upright = np.abs(axis[:, 1]) / np.maximum(length, 1e-9) > min_vertical
    return (width / np.maximum(length, 1e-9))[upright]


def plot_class_balance(arrays, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(TARGET_CLASSES))
    for i, split in enumerate(SPLITS):
        mask = arrays["split"] == i
        counts = [int((mask & (arrays["label"] == HAGRID_CLASSES.index(c))).sum()) for c in TARGET_CLASSES]
        ax.bar(x + (i - 1) * 0.27, counts, width=0.27, label=split)
    ax.set_xticks(x, TARGET_CLASSES)
    ax.set_yscale("log")
    ax.set_ylabel("hands (log scale)")
    ax.set_title("HaGRIDv2 target classes per subject-disjoint split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_palm_ratio(arrays, path):
    open_hand = np.isin(arrays["label"], [HAGRID_CLASSES.index(c) for c in OPEN_HAND_CLASSES])
    ratios = palm_ratio(arrays["landmarks"][open_hand])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ratios, bins=np.geomspace(0.2, 2.5, 80))
    ax.set_xscale("log")
    ax.set_xlabel("palm width / palm length, in raw normalised coordinates (log scale)")
    ax.set_ylabel("upright open hands")
    ax.set_title("Upright open hands: palm shape is multi-modal because image aspect ratios are mixed")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "processed" / "hagrid_landmarks.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "figures")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    arrays = load_cache(args.cache)
    plot_class_balance(arrays, args.out / "class_balance.png")
    plot_palm_ratio(arrays, args.out / "palm_ratio.png")
    print(f"Saved figures to {args.out}")


if __name__ == "__main__":
    main()
