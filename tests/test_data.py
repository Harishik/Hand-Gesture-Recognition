import json

import numpy as np
import pytest

from hgr.features import FEATURE_DIM, featurize, normalize_landmarks
from hgr.hagrid import CLASS_TO_ID, load_cache, load_hagrid, parse_annotations, save_cache
from hgr.splits import group_split, is_subject_disjoint, make_subject_disjoint, subject_overlap


def _hand(rng):
    return rng.random((21, 2)).tolist()


@pytest.fixture
def annotations():
    rng = np.random.default_rng(0)
    return {
        "img-a": {  # one-handed gesture plus a second, gesture-free hand
            "bboxes": [[0.1, 0.1, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]],
            "labels": ["like", "no_gesture"],
            "hand_landmarks": [_hand(rng), _hand(rng)],
            "united_bbox": None,
            "united_label": None,
            "user_id": "u1",
            "meta": {"age": [30.5, 22.0], "gender": ["male", "female"], "race": [None]},
        },
        "img-b": {  # MediaPipe failed on the hand: no landmarks
            "bboxes": [[0.3, 0.3, 0.1, 0.1]],
            "labels": ["like"],
            "hand_landmarks": [[]],
            "united_bbox": None,
            "united_label": None,
            "user_id": "u2",
            "meta": {},
        },
    }


def test_parse_keeps_hands_with_landmarks_and_counts_dropped(annotations):
    cols, dropped = parse_annotations(annotations, "val", "like")
    assert dropped == 1
    assert cols["label"] == [CLASS_TO_ID["like"], CLASS_TO_ID["no_gesture"]]
    assert cols["file_class"] == [CLASS_TO_ID["like"]] * 2
    assert cols["split"] == [1, 1]
    assert cols["age"] == [30.5, 30.5]
    assert cols["gender"] == [1, 1]
    assert cols["race"] == [-1, -1]  # race recorded as null


def test_load_and_cache_roundtrip(tmp_path, annotations):
    for split in ("train", "val", "test"):
        (tmp_path / "annotations" / split).mkdir(parents=True)
    with open(tmp_path / "annotations" / "train" / "like.json", "w") as f:
        json.dump(annotations, f)

    arrays, dropped = load_hagrid(tmp_path, workers=1)
    assert dropped == 1
    assert arrays["landmarks"].shape == (2, 21, 2)
    assert arrays["image_id"].tolist() == [b"img-a", b"img-a"]

    save_cache(arrays, tmp_path / "cache.npz")
    loaded = load_cache(tmp_path / "cache.npz")
    np.testing.assert_array_equal(loaded["landmarks"], arrays["landmarks"])
    assert set(loaded) == set(arrays)


def test_overlap_is_detected_and_resolved_by_majority_split():
    subjects = np.array(["a", "a", "a", "b", "c", "c"])
    split = np.array([0, 2, 2, 1, 0, 0])
    assert subject_overlap(subjects, split)[(0, 2)] == 1
    fixed = make_subject_disjoint(subjects, split)
    assert fixed.tolist() == [2, 2, 2, 1, 0, 0]
    assert is_subject_disjoint(subjects, fixed)


def test_group_split_is_subject_disjoint_and_seeded():
    subjects = np.repeat(np.arange(100), 5)
    split = group_split(subjects, (0.8, 0.1, 0.1), seed=1)
    assert is_subject_disjoint(subjects, split)
    assert np.bincount(split).tolist() == [400, 50, 50]
    np.testing.assert_array_equal(split, group_split(subjects, (0.8, 0.1, 0.1), seed=1))


def test_features_invariant_to_translation_and_scale():
    lm = np.random.default_rng(0).random((4, 21, 2)).astype(np.float32)
    moved = lm * 0.5 + 0.2
    np.testing.assert_allclose(featurize(lm), featurize(moved), atol=1e-4)
    assert featurize(lm).shape == (4, FEATURE_DIM)


def test_features_keep_orientation():
    lm = np.random.default_rng(1).random((1, 21, 2)).astype(np.float32)
    upside_down = 1.0 - lm  # 180 degree rotation: thumbs up vs thumbs down must stay distinguishable
    assert not np.allclose(featurize(lm), featurize(upside_down), atol=1e-3)


def test_aspect_makes_units_isotropic():
    # A square in pixels on a 16:9 frame is wider than tall in normalised coordinates.
    lm = np.zeros((1, 21, 2), dtype=np.float32)
    lm[0, 9] = (0.0, 0.16)  # wrist -> middle MCP: vertical, length 0.16 * H
    lm[0, 5] = (0.09, 0.0)  # 0.09 * W = 0.16 * H when W/H = 16/9
    pts = normalize_landmarks(lm, aspect=16 / 9)
    assert np.linalg.norm(pts[0, 5]) == pytest.approx(1.0, rel=1e-3)
