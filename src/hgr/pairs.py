"""Two-handed gestures: features of a pair of hands, and pairing of HaGRID hands that share an image.

A two-handed gesture (hand heart, X sign, timeout, ...) is only defined by both hands together, so a second classifier
looks at the pair: both hands' own features, plus their points in a shared frame (centred between the wrists and
scaled by mean palm length) so their relative position and pose are visible.
"""

import numpy as np

from hgr.augment import augment_points, sample_aspects
from hgr.features import FEATURE_DIM, MIDDLE_MCP, NUM_LANDMARKS, WRIST, features_from_points, normalize_landmarks
from hgr.labels import HAGRID_CLASSES, REJECT, TWO_HANDED

PAIR_CLASSES = (*TWO_HANDED, REJECT)
PAIR_FEATURE_DIM = 2 * NUM_LANDMARKS * 2 + 2 * FEATURE_DIM + 3


def isotropic(pairs, aspect):
    """(N, 2, 21, 2) image-normalised landmarks -> same, with x rescaled by the image aspect (width / height)."""
    pts = np.asarray(pairs, dtype=np.float32).copy()
    pts[..., 0] *= np.asarray(aspect, dtype=np.float32).reshape(-1, 1, 1)
    return pts


def pair_features_from_points(pts):
    """(N, 2, 21, 2) isotropic points of two hands -> (N, PAIR_FEATURE_DIM) features. Hand order doesn't matter."""
    pts = np.asarray(pts, dtype=np.float32)
    swap = pts[:, 0, WRIST, 0] > pts[:, 1, WRIST, 0]  # left-most wrist first
    pts = np.where(swap[:, None, None, None], pts[:, ::-1], pts)

    center = pts[:, :, WRIST].mean(axis=1)  # (N, 2)
    palm = np.linalg.norm(pts[:, :, MIDDLE_MCP] - pts[:, :, WRIST], axis=-1).mean(axis=1)  # (N,)
    scale = np.maximum(palm, 1e-6)[:, None, None, None]
    shared = (pts - center[:, None, None, :]) / scale

    per_hand = [features_from_points(normalize_landmarks(pts[:, h])) for h in range(2)]
    wrist_vec = shared[:, 1, WRIST] - shared[:, 0, WRIST]
    relative = np.concatenate([wrist_vec, np.linalg.norm(wrist_vec, axis=-1, keepdims=True)], axis=1)
    return np.concatenate([shared.reshape(len(pts), -1), *per_hand, relative], axis=1).astype(np.float32)


def pair_features(pairs, aspect=1.0):
    """(N, 2, 21, 2) raw landmarks of two hands -> features."""
    return pair_features_from_points(isotropic(pairs, aspect))


def augmented_pair_features(pairs, rng, augment=True, max_rotation_deg=20.0):
    """Pair features with aspects from the HaGRID prior and (optionally) mirror / rotation / stretch / jitter.

    The same transform is applied to both hands so their relative pose stays consistent.
    """
    pts = isotropic(pairs, sample_aspects(len(pairs), rng))
    if augment:
        n = len(pts)
        center = pts[:, :, WRIST].mean(axis=1)[:, None, None, :]
        flat = (pts - center).reshape(n, 2 * NUM_LANDMARKS, 2)
        flat = augment_points(flat, rng, max_rotation_deg=max_rotation_deg, noise=0.004)
        pts = flat.reshape(n, 2, NUM_LANDMARKS, 2)
    return pair_features_from_points(pts)


def build_pairs(arrays, split):
    """Images of `split` with exactly two landmark-annotated hands. Returns ((N, 2, 21, 2) landmarks, (N,) labels).

    The label is the two-handed gesture if HaGRID gives it to both hands, otherwise no_gesture (two hands in view
    that don't form a two-handed gesture, e.g. one gesture hand next to a relaxed one).
    """
    idx = np.flatnonzero(arrays["split"] == split)
    idx = idx[np.argsort(arrays["image_id"][idx], kind="stable")]
    ids = arrays["image_id"][idx]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    counts = np.diff(np.r_[starts, len(ids)])
    first = idx[starts[counts == 2]]
    second = idx[starts[counts == 2] + 1]

    two_handed_ids = {HAGRID_CLASSES.index(c): i for i, c in enumerate(TWO_HANDED)}
    la, lb = arrays["label"][first], arrays["label"][second]
    reject = PAIR_CLASSES.index(REJECT)
    y = np.array([two_handed_ids.get(a, reject) if a == b else reject for a, b in zip(la, lb, strict=True)])
    landmarks = np.stack([arrays["landmarks"][first], arrays["landmarks"][second]], axis=1)
    return landmarks, y.astype(np.int64)
