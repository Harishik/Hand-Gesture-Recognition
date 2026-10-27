"""Gesture label vocabulary shared by the datasets and models."""

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
