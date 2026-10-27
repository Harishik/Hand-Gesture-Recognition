import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hgr.augment import augment_points, augmented_features, sample_aspects  # noqa: E402
from hgr.features import FEATURE_DIM, normalize_landmarks  # noqa: E402
from hgr.hagrid import ASPECT_PRIOR  # noqa: E402
from hgr.metrics import classification_metrics, expected_calibration_error  # noqa: E402
from hgr.model import GestureClassifier, GestureMLP  # noqa: E402
from hgr.train import balanced_epoch, balanced_macro_f1, tune_reject_bias  # noqa: E402

CLASSES = ("a", "b", "no_gesture")


def test_aspects_come_from_prior():
    aspects = sample_aspects(1000, np.random.default_rng(0))
    prior = np.array(list(ASPECT_PRIOR))
    assert np.isclose(aspects[:, None], prior[None, :], atol=1e-5).any(axis=1).all()


def test_augmentation_keeps_shape_and_mirror_flips_x():
    rng = np.random.default_rng(0)
    pts = normalize_landmarks(rng.random((8, 21, 2)))
    mirrored = augment_points(pts, rng, mirror_p=1.0, max_rotation_deg=0, stretch=0, noise=0)
    np.testing.assert_allclose(mirrored[..., 0], -pts[..., 0], atol=1e-6)
    np.testing.assert_allclose(mirrored[..., 1], pts[..., 1], atol=1e-6)
    assert augmented_features(rng.random((8, 21, 2)), rng).shape == (8, FEATURE_DIM)


def test_classifier_roundtrip_and_reject_bias(tmp_path):
    model = GestureMLP(len(CLASSES), hidden=16, depth=1)
    clf = GestureClassifier(model, CLASSES, temperature=2.0, reject_bias=0.5)
    x = np.random.default_rng(0).normal(size=(5, FEATURE_DIM)).astype(np.float32)
    clf.save(tmp_path / "m.pt")
    loaded = GestureClassifier.load(tmp_path / "m.pt")
    np.testing.assert_allclose(loaded.predict_proba(x), clf.predict_proba(x), atol=1e-6)
    np.testing.assert_allclose(loaded.predict_proba(x).sum(1), 1.0, atol=1e-6)

    much_more_rejection = GestureClassifier(model, CLASSES, reject_bias=100.0)
    assert (much_more_rejection.predict(x) == CLASSES.index("no_gesture")).all()


def test_balanced_epoch_draws_equal_counts():
    y = np.array([0] * 5 + [1] * 50 + [2] * 500)
    idx = balanced_epoch(y, 20, np.random.default_rng(0))
    assert np.bincount(y[idx]).tolist() == [20, 20, 20]


def test_tune_reject_bias_prefers_rejecting_when_rejection_is_underweighted():
    # The model under-predicts class 2 (rejection): a positive bias on it should be chosen.
    y = np.array([0, 1, 2, 2])
    logits = np.array([[2.0, 0, 0], [0, 2.0, 0], [1.0, 0, 0.5], [0, 1.0, 0.5]])
    bias, _ = tune_reject_bias(logits, y, temperature=1.0, reject_index=2)
    assert bias > 0.5
    z = logits.copy()
    z[:, 2] += bias
    assert balanced_macro_f1(y, z.argmax(1), 3) == pytest.approx(1.0)


def test_metrics_and_ece():
    y = np.array([0, 0, 1, 2])
    m = classification_metrics(y, np.array([0, 1, 1, 2]), CLASSES)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class"]["b"]["precision"] == pytest.approx(0.5)
    assert m["per_class"]["a"]["recall"] == pytest.approx(0.5)
    probs = np.eye(3)[[0, 0, 1, 2]]
    assert expected_calibration_error(probs, np.array([0, 0, 1, 2])) == pytest.approx(0.0)


def test_peace_inverted_counts_as_peace_in_core6_and_other_gestures_are_rejected():
    from hgr.labels import GESTURE_SETS, HAGRID_CLASSES
    from hgr.train import select

    classes = GESTURE_SETS["core6"]["classes"]
    names = ["peace", "peace_inverted", "stop", "no_gesture"]
    arrays = {
        "label": np.array([HAGRID_CLASSES.index(n) for n in names]),
        "split": np.zeros(4, dtype=np.int8),
        "landmarks": np.zeros((4, 21, 2), dtype=np.float32),
    }
    _, y = select(arrays, 0, "all", "core6")
    assert [classes[i] for i in y] == ["peace", "peace", "no_gesture", "no_gesture"]
    _, y = select(arrays, 0, "no_gesture", "core6")
    assert [classes[i] for i in y] == ["peace", "peace", "no_gesture"]

    full = GESTURE_SETS["hagrid"]["classes"]
    _, y = select(arrays, 0, "all", "hagrid")
    assert [full[i] for i in y] == ["peace", "peace_inverted", "stop", "no_gesture"]
