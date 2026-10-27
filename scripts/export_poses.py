"""Export one representative pose per gesture for the web UI's pose glyphs.

    python scripts/export_poses.py [--out web/models/poses.json]

For each class, hands are normalised (wrist at origin, palm length 1, aspect from the HaGRID prior), mirrored so the
thumb is on the same side, and the medoid (the real sample closest to the class mean) is exported. Two-handed
gestures get a medoid pair in the shared two-hand frame.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from hgr.features import MIDDLE_MCP, WRIST, normalize_landmarks
from hgr.hagrid import load_cache
from hgr.labels import HAGRID_CLASSES, ONE_HANDED, TWO_HANDED
from hgr.pairs import build_pairs, isotropic

REPO_ROOT = Path(__file__).resolve().parents[1]
ASPECT = 3 / 4  # the most common HaGRID aspect (docs/DATA.md)


def canonical(pts):
    """Mirror hands so the thumb tip is left of the little-finger base (one consistent handedness)."""
    pts = pts.copy()
    flip = pts[:, 4, 0] > pts[:, 17, 0]
    pts[flip, :, 0] *= -1
    return pts


def medoid(samples):
    flat = samples.reshape(len(samples), -1)
    return samples[np.argmin(np.linalg.norm(flat - flat.mean(0), axis=1))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "processed" / "hagrid_landmarks.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "web" / "models" / "poses.json")
    parser.add_argument("--per-class", type=int, default=3000)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    arrays = load_cache(args.cache)
    train = arrays["split"] == 0
    poses = {}
    for name in ONE_HANDED:
        idx = np.flatnonzero(train & (arrays["label"] == HAGRID_CLASSES.index(name)))
        idx = rng.choice(idx, min(args.per_class, len(idx)), replace=False)
        pts = canonical(normalize_landmarks(arrays["landmarks"][idx], ASPECT))
        poses[name] = {"hands": [np.round(medoid(pts), 3).tolist()]}

    pairs, y = build_pairs(arrays, 0)
    for i, name in enumerate(TWO_HANDED):
        sel = pairs[y == i]
        sel = sel[rng.choice(len(sel), min(args.per_class, len(sel)), replace=False)]
        pts = isotropic(sel, ASPECT)
        swap = pts[:, 0, WRIST, 0] > pts[:, 1, WRIST, 0]
        pts = np.where(swap[:, None, None, None], pts[:, ::-1], pts)
        center = pts[:, :, WRIST].mean(axis=1)[:, None, None, :]
        palm = np.linalg.norm(pts[:, :, MIDDLE_MCP] - pts[:, :, WRIST], axis=-1).mean(axis=1)[:, None, None, None]
        shared = (pts - center) / np.maximum(palm, 1e-6)
        poses[name] = {"hands": np.round(medoid(shared), 3).tolist()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(poses))
    print(f"Saved {len(poses)} poses to {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
