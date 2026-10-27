"""Training-time augmentation of hand landmarks.

HaGRID annotations don't record image size, so each sample gets an aspect ratio drawn from the measured prior
(see docs/DATA.md). The remaining augmentations act on wrist-centred, palm-scaled points.
"""

import numpy as np

from hgr.features import features_from_points, normalize_landmarks
from hgr.hagrid import ASPECT_PRIOR

_ASPECTS = np.array(list(ASPECT_PRIOR), dtype=np.float32)
_ASPECT_P = np.array(list(ASPECT_PRIOR.values()), dtype=np.float64) / sum(ASPECT_PRIOR.values())


def sample_aspects(n, rng):
    return rng.choice(_ASPECTS, size=n, p=_ASPECT_P)


def augment_points(pts, rng, mirror_p=0.5, max_rotation_deg=20.0, stretch=0.08, noise=0.02):
    """Random mirror (left <-> right hand), small rotation, small anisotropic stretch and per-point jitter.

    Rotation stays small: `like` and `dislike` differ by orientation.
    """
    pts = pts.copy()
    n = len(pts)
    flip = rng.random(n) < mirror_p
    pts[flip, :, 0] *= -1

    theta = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg, n)).astype(np.float32)
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)  # (n, 2, 2)
    pts = np.einsum("nij,nkj->nki", rot, pts)

    pts *= rng.uniform(1 - stretch, 1 + stretch, (n, 1, 2)).astype(np.float32)
    pts += rng.normal(0.0, noise, pts.shape).astype(np.float32)
    return pts


def augmented_features(landmarks, rng, augment=True, max_rotation_deg=20.0):
    """Raw HaGRID landmarks -> features, with aspects sampled from the prior and optional augmentation."""
    pts = normalize_landmarks(landmarks, sample_aspects(len(landmarks), rng))
    if augment:
        pts = augment_points(pts, rng, max_rotation_deg=max_rotation_deg)
    return features_from_points(pts)
