import numpy as np

from hgr.hagrid import CLASS_TO_ID
from hgr.labels import GESTURE_SETS, ONE_HANDED, TWO_HANDED
from hgr.pairs import PAIR_CLASSES, PAIR_FEATURE_DIM, augmented_pair_features, build_pairs, pair_features


def test_vocabulary_covers_all_hagrid_gestures():
    assert len(ONE_HANDED) == 26 and len(TWO_HANDED) == 7
    assert GESTURE_SETS["hagrid"]["classes"][-1] == "no_gesture"
    assert PAIR_CLASSES == (*TWO_HANDED, "no_gesture")


def test_pair_features_ignore_hand_order_and_have_fixed_size():
    rng = np.random.default_rng(0)
    pairs = rng.random((3, 2, 21, 2)).astype(np.float32)
    f = pair_features(pairs, 4 / 3)
    assert f.shape == (3, PAIR_FEATURE_DIM)
    np.testing.assert_allclose(f, pair_features(pairs[:, ::-1], 4 / 3), atol=1e-5)
    assert augmented_pair_features(pairs, rng).shape == (3, PAIR_FEATURE_DIM)


def test_build_pairs_labels_two_handed_gestures_and_skips_single_hands():
    label = [CLASS_TO_ID[c] for c in ("hand_heart", "hand_heart", "like", "no_gesture", "fist")]
    arrays = {
        "split": np.zeros(5, dtype=np.int8),
        "image_id": np.array([b"a", b"a", b"b", b"b", b"c"]),
        "label": np.array(label),
        "landmarks": np.zeros((5, 21, 2), dtype=np.float32),
    }
    landmarks, y = build_pairs(arrays, 0)
    assert landmarks.shape == (2, 2, 21, 2)
    assert [PAIR_CLASSES[i] for i in y] == ["hand_heart", "no_gesture"]
