"""Hand-landmark feature pipeline.

Input: MediaPipe's 21 hand landmarks as (N, 21, 2) arrays of image-normalised (x, y) coordinates.
Output: (N, FEATURE_DIM) float32 vectors that are invariant to hand position and size.

Rotation is deliberately *not* normalised away: `like` and `dislike` (thumbs up / down) differ only by orientation.
"""

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
FINGERTIPS = (4, 8, 12, 16, 20)
# Joint chains from the wrist to each fingertip: thumb, index, middle, ring, pinky.
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)

_TIP_PAIRS = np.array([(a, b) for i, a in enumerate(FINGERTIPS) for b in FINGERTIPS[i + 1 :]])
_JOINT_TRIPLES = np.array([chain[i : i + 3] for chain in FINGER_CHAINS for i in range(3)])

FEATURE_DIM = NUM_LANDMARKS * 2 + len(_JOINT_TRIPLES) + len(_TIP_PAIRS)  # 42 + 15 + 10 = 67


def normalize_landmarks(landmarks, aspect=1.0):
    """Make landmarks translation- and scale-invariant.

    `aspect` is the image width / height. Landmarks are normalised by width and height separately, so x is
    rescaled by the aspect to get isotropic units before measuring shapes. Then the wrist moves to the origin
    and coordinates are divided by palm length (wrist -> middle-finger MCP), which barely changes between
    gestures, unlike the overall hand extent.
    """
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, NUM_LANDMARKS, 2).copy()
    pts[..., 0] *= np.asarray(aspect, dtype=np.float32).reshape(-1, 1)
    pts -= pts[:, WRIST : WRIST + 1]
    palm = np.linalg.norm(pts[:, MIDDLE_MCP], axis=-1, keepdims=True)
    pts /= np.maximum(palm, 1e-6)[:, :, None]
    return pts


def joint_angles(pts):
    """Bend angle (radians) at the 3 joints of each finger: (N, 15). 0 = straight."""
    a, b, c = (pts[:, _JOINT_TRIPLES[:, i]] for i in range(3))
    v1, v2 = b - a, c - b
    cos = np.sum(v1 * v2, axis=-1) / np.maximum(np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1), 1e-6)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def fingertip_distances(pts):
    """Pairwise distances between the 5 fingertips: (N, 10)."""
    return np.linalg.norm(pts[:, _TIP_PAIRS[:, 0]] - pts[:, _TIP_PAIRS[:, 1]], axis=-1)


def featurize(landmarks, aspect=1.0):
    """(N, 21, 2) raw landmarks -> (N, FEATURE_DIM) float32 features."""
    pts = normalize_landmarks(landmarks, aspect)
    feats = np.concatenate([pts.reshape(len(pts), -1), joint_angles(pts), fingertip_distances(pts)], axis=1)
    return feats.astype(np.float32)
