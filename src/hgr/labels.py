"""Gesture label vocabulary shared by the datasets and models."""

# All 34 HaGRIDv2 classes (33 gestures + no_gesture), in a fixed order used for integer label ids in caches.
HAGRID_CLASSES = (
    "call", "dislike", "fist", "four", "grabbing", "grip", "hand_heart", "hand_heart2", "holy", "like",
    "little_finger", "middle_finger", "mute", "no_gesture", "ok", "one", "palm", "peace", "peace_inverted", "point",
    "rock", "stop", "stop_inverted", "take_picture", "three", "three2", "three3", "three_gun", "thumb_index",
    "thumb_index2", "timeout", "two_up", "two_up_inverted", "xsign",
)  # fmt: skip

# Classes the project targets, named as in HaGRIDv2, plus the rejection class.
TARGET_CLASSES = ("palm", "fist", "peace", "ok", "like", "dislike", "no_gesture")

# Folder names of the hand-captured personal test set -> HaGRIDv2 class names.
PERSONAL_TO_HAGRID = {
    "hello": "palm",
    "fist": "fist",
    "peace": "peace",
    "okay": "ok",
    "thumbs_up": "like",
    "thumbs_down": "dislike",
}

CLASS_TO_INDEX = {name: i for i, name in enumerate(TARGET_CLASSES)}

# HaGRID classes that count as a target class. HaGRID's "_inverted" variants are the same hand shape with the back
# of the hand towards the camera (not upside down); a V sign is a peace sign either way.
HAGRID_ALIASES = {"peace_inverted": "peace"}
