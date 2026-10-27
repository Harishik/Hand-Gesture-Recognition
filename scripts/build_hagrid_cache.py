"""Parse the HaGRIDv2 annotations into one cached array file and report dataset statistics.

python scripts/build_hagrid_cache.py [--root data/raw/hagrid] [--out data/processed/hagrid_landmarks.npz]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from hgr.hagrid import SPLITS, load_hagrid, save_cache
from hgr.labels import HAGRID_CLASSES, TARGET_CLASSES
from hgr.splits import is_subject_disjoint, make_subject_disjoint, subject_overlap

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "raw" / "hagrid")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed" / "hagrid_landmarks.npz")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    start = time.perf_counter()
    arrays, dropped = load_hagrid(args.root, workers=args.workers)
    # HaGRID's official split shares a few subjects between train and test; keep it, but use a disjoint one.
    arrays["split_official"] = arrays["split"]
    arrays["split"] = make_subject_disjoint(arrays["user_id"], arrays["split"])
    moved = int((arrays["split"] != arrays["split_official"]).sum())
    save_cache(arrays, args.out)
    n = len(arrays["label"])
    print(f"Cached {n:,} hands ({dropped:,} dropped: no landmarks) in {time.perf_counter() - start:.0f}s -> {args.out}")

    split = arrays["split"]
    users = arrays["user_id"]
    print("\nHands / images / subjects per split:")
    for i, name in enumerate(SPLITS):
        mask = split == i
        print(
            f"  {name:5s} {mask.sum():>9,} / {len(np.unique(arrays['image_id'][mask])):>9,}"
            f" / {len(np.unique(users[mask])):>7,}"
        )
    official = {(SPLITS[a], SPLITS[b]): k for (a, b), k in subject_overlap(users, arrays["split_official"]).items()}
    print(f"\nSubjects shared between official splits: {official}")
    print(f"Moved {moved} hand(s) to make splits subject-disjoint; disjoint now: {is_subject_disjoint(users, split)}")

    print("\nTarget classes (hands per split):")
    for name in TARGET_CLASSES:
        mask = arrays["label"] == HAGRID_CLASSES.index(name)
        counts = [int((mask & (split == i)).sum()) for i in range(len(SPLITS))]
        print(f"  {name:11s} " + " ".join(f"{c:>9,}" for c in counts))


if __name__ == "__main__":
    main()
