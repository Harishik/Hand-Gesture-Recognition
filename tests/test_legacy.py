import numpy as np
import pytest

pytest.importorskip("skimage")

from hgr.labels import PERSONAL_TO_HAGRID, TARGET_CLASSES  # noqa: E402
from hgr.legacy.ssim_matcher import (  # noqa: E402
    DEFAULT_DATA_DIR,
    GESTURE_DIRS,
    crop_hand,
    load_templates,
    preprocess,
    recognize_gesture,
)


def test_crop_is_padded_and_clamped_to_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_hand(frame, (-10, 90, 50, 30), padding=5)
    assert crop.shape == (15, 45, 3)  # x: 0..45, y: 85..100


def test_crop_outside_frame_returns_none():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop_hand(frame, (300, 300, 10, 10), padding=0) is None


def test_templates_load_for_every_gesture():
    templates = load_templates(DEFAULT_DATA_DIR)
    assert set(templates) == set(GESTURE_DIRS)
    assert all(len(images) > 0 for images in templates.values())
    assert all(img.shape == (50, 50) for images in templates.values() for img in images)


def test_template_matches_itself():
    templates = load_templates(DEFAULT_DATA_DIR)
    query = templates["Peace"][0]
    gesture, score = recognize_gesture(query, templates)
    assert gesture == "Peace"
    assert score == pytest.approx(1.0)


def test_preprocess_normalises_to_unit_range():
    img = np.full((80, 60, 3), 255, dtype=np.uint8)
    out = preprocess(img)
    assert out.shape == (50, 50)
    assert out.max() == pytest.approx(1.0)


def test_personal_folders_map_to_target_classes():
    assert set(PERSONAL_TO_HAGRID) == set(GESTURE_DIRS.values())
    assert set(PERSONAL_TO_HAGRID.values()) <= set(TARGET_CLASSES)
