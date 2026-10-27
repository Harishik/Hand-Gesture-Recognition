"""Gesture label vocabulary shared by the datasets and models."""

# All 34 HaGRIDv2 classes (33 gestures + no_gesture), in a fixed order used for integer label ids in caches.
HAGRID_CLASSES = (
    "call", "dislike", "fist", "four", "grabbing", "grip", "hand_heart", "hand_heart2", "holy", "like",
    "little_finger", "middle_finger", "mute", "no_gesture", "ok", "one", "palm", "peace", "peace_inverted", "point",
    "rock", "stop", "stop_inverted", "take_picture", "three", "three2", "three3", "three_gun", "thumb_index",
    "thumb_index2", "timeout", "two_up", "two_up_inverted", "xsign",
)  # fmt: skip

# Gestures made with both hands together; HaGRID labels each of the two hands with the gesture name.
TWO_HANDED = ("hand_heart", "hand_heart2", "holy", "take_picture", "thumb_index2", "timeout", "xsign")
ONE_HANDED = tuple(c for c in HAGRID_CLASSES if c not in TWO_HANDED and c != "no_gesture")  # 26 gestures

REJECT = "no_gesture"

# Per-hand class sets a model can be trained on (the rejection class always last). Each set has its own aliases:
# HaGRID classes merged into one of its classes. Everything else becomes no_gesture.
GESTURE_SETS = {
    # All 26 one-handed HaGRIDv2 gestures. Two-handed gestures are recognised by the pair model instead.
    "hagrid": {"classes": (*ONE_HANDED, REJECT), "aliases": {}},
    # The original class project's six gestures. HaGRID's "_inverted" variants are the same hand shape with the back
    # of the hand towards the camera (not upside down); a V sign is a peace sign either way.
    "core6": {
        "classes": ("palm", "fist", "peace", "ok", "like", "dislike", REJECT),
        "aliases": {"peace_inverted": "peace"},
    },
}

# The original six-gesture vocabulary (used by the personal test set and the clip recorder).
TARGET_CLASSES = GESTURE_SETS["core6"]["classes"]

# Classes that differ only in which side of the hand faces the camera. Counted as equivalent in "lenient" scoring.
ORIENTATION_EQUIVALENT = {
    "peace_inverted": "peace",
    "stop_inverted": "stop",
    "two_up_inverted": "two_up",
}

# Folder names of the hand-captured personal test set -> HaGRIDv2 class names.
PERSONAL_TO_HAGRID = {
    "hello": "palm",
    "fist": "fist",
    "peace": "peace",
    "okay": "ok",
    "thumbs_up": "like",
    "thumbs_down": "dislike",
}
