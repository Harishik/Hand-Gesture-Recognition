"""HaGRIDv2 annotation loader: one record per annotated hand that has MediaPipe landmarks.

Annotation files live at `<root>/annotations/{train,val,test}/<class>.json`. Each maps an image id to:
`bboxes`, `labels` and `hand_landmarks` (one entry per hand; landmarks may be an empty list when MediaPipe
failed), `user_id`, `united_bbox`/`united_label` (two-handed gestures) and `meta` (age/gender/race of the faces
in the image, auto-annotated and not aligned with the hands).

The annotations record neither image size nor handedness.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from hgr.labels import HAGRID_CLASSES

SPLITS = ("train", "val", "test")
GENDERS = ("female", "male")
RACES = ("White", "Black", "Asian", "Indian")
CLASS_TO_ID = {name: i for i, name in enumerate(HAGRID_CLASSES)}

# Image width / height distribution, since the annotations don't record image size. Measured on 400 random images
# of the public 384p HaGRID sample (hf.co/datasets/cj-mills/hagrid-sample-30k-384p), matched to these annotations
# by image id. Using the true aspect collapses the palm width/length ratio of upright open hands onto ~0.70 for both
# portrait and landscape images (raw: 0.95 vs 0.51), matching MediaPipe's metric world landmarks.
ASPECT_PRIOR = {3 / 4: 0.667, 4 / 3: 0.152, 9 / 16: 0.095, 16 / 9: 0.068, 1.0: 0.018}

_FIELDS = {
    "landmarks": np.float32,  # (N, 21, 2) image-normalised x, y
    "bbox": np.float32,  # (N, 4) normalised [x, y, w, h]
    "label": np.int16,  # index into HAGRID_CLASSES
    "file_class": np.int16,  # class of the annotation file the hand came from
    "split": np.int8,  # index into SPLITS
    "user_id": "S64",
    "image_id": "S36",
    "two_handed": np.bool_,  # part of a two-handed gesture (united_label set)
    "age": np.float32,  # first face in the image; NaN if missing
    "gender": np.int8,  # index into GENDERS, -1 if missing
    "race": np.int8,  # index into RACES, -1 if missing
}


def annotation_files(root):
    """All (split, class, path) annotation files under root, skipping stray notebook checkpoint copies."""
    root = Path(root)
    files = []
    for split in SPLITS:
        for path in sorted((root / "annotations" / split).glob("*.json")):
            if path.stem not in CLASS_TO_ID:
                raise ValueError(f"Unknown HaGRID class file: {path}")
            files.append((split, path.stem, path))
    return files


def _first(values, lookup=None, missing=-1):
    if not values or values[0] is None:
        return missing
    return values[0] if lookup is None else lookup.get(values[0], missing)


def parse_annotations(annotations, split, file_class):
    """Turn one annotation dict into per-hand column lists. Returns (columns, number of hands without landmarks)."""
    cols = {name: [] for name in _FIELDS}
    gender_ids = {g: i for i, g in enumerate(GENDERS)}
    race_ids = {r: i for i, r in enumerate(RACES)}
    dropped = 0
    for image_id, ann in annotations.items():
        meta = ann.get("meta") or {}
        age = _first(meta.get("age"), missing=np.nan)
        gender = _first(meta.get("gender"), gender_ids)
        race = _first(meta.get("race"), race_ids)
        two_handed = ann.get("united_label") is not None
        for bbox, label, landmarks in zip(ann["bboxes"], ann["labels"], ann["hand_landmarks"], strict=True):
            if len(landmarks) != 21:
                dropped += 1
                continue
            cols["landmarks"].append(landmarks)
            cols["bbox"].append(bbox)
            cols["label"].append(CLASS_TO_ID[label])
            cols["file_class"].append(CLASS_TO_ID[file_class])
            cols["split"].append(SPLITS.index(split))
            cols["user_id"].append(ann["user_id"])
            cols["image_id"].append(image_id)
            cols["two_handed"].append(two_handed)
            cols["age"].append(age)
            cols["gender"].append(gender)
            cols["race"].append(race)
    return cols, dropped


def _load_file(args):
    split, file_class, path = args
    with open(path, encoding="utf-8") as f:
        cols, dropped = parse_annotations(json.load(f), split, file_class)
    arrays = {name: np.array(values, dtype=_FIELDS[name]) for name, values in cols.items()}
    arrays["landmarks"] = arrays["landmarks"].reshape(-1, 21, 2)
    arrays["bbox"] = arrays["bbox"].reshape(-1, 4)
    return arrays, dropped


def load_hagrid(root, workers=None):
    """Load every annotation file into one dict of column arrays. Returns (arrays, number of hands dropped)."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_load_file, annotation_files(root)))
    arrays = {name: np.concatenate([r[0][name] for r in results]) for name in _FIELDS}
    return arrays, sum(r[1] for r in results)


def save_cache(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, classes=np.array(HAGRID_CLASSES), **arrays)


def load_cache(path):
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    if tuple(arrays.pop("classes")) != HAGRID_CLASSES:
        raise ValueError(f"{path} was built with a different class list; rebuild it")
    return arrays
