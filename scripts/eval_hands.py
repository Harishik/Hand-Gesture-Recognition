"""Cross-dataset evaluation on HANDS (Nuzzi et al. 2021, CC BY 4.0) with models trained only on HaGRIDv2.

    python scripts/eval_hands.py [--root data/raw/hands] [--every 5] [--out reports/cross_dataset_hands.json]

HANDS: 5 subjects, Kinect v2 RGB, 12 single-hand gestures (each with the right and left hand) + 3 two-hand gestures,
one bounding box per hand per frame. Each frame goes through the same MediaPipe landmarker as the live app. The
detected hand that best overlaps the annotated box is classified by the HaGRID-trained one-hand model; two-hand frames
use the pair model. HANDS gestures with no HaGRID equivalent measure rejection: a correct outcome is no_gesture.
"""

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from hgr.features import featurize
from hgr.landmarker import HandLandmarker
from hgr.model import GestureClassifier
from hgr.pairs import pair_features

REPO_ROOT = Path(__file__).resolve().parents[1]

# HANDS gesture -> acceptable HaGRIDv2 classes, decided from a contact sheet of HANDS frames (Italian counting signs).
# None: no HaGRID equivalent; reported descriptively (what the model predicts), not scored.
ONE_HAND_MAP = {
    "Punch": {"fist"},
    "One": {"one"},
    "Two": {"peace", "peace_inverted"},
    "Three": {"three", "three2", "three3"},
    "Four": {"four"},
    "Five": {"palm"},
    "Span": {"call"},
    "Six": None,
    "Seven": None,
    "Eight": None,
    "Nine": None,
    "Horiz": None,
}
TWO_HAND_MAP = {"XSign": {"xsign"}, "TimeOut": {"timeout"}, "Collab": None}


def parse_annotations(path):
    """Yields (image path, gesture column, [x, y, w, h]) for every annotated hand.

    Note: SubjectN.txt names its images "Subject{N+1}" (an offset in the release); frames are matched by file name
    within SubjectN.zip, which a contact sheet of annotated crops confirms is the correct pairing.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    box = re.compile(r"\[(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+)\]")
    for line in lines[1:]:
        boxes = box.findall(line)
        rgb = line.split(",")[0]
        for col, b in zip(header[2:], boxes, strict=True):
            b = [float(v) for v in b]
            if any(b):
                yield rgb, col, b


def base_gesture(column):
    return column.split("_")[0]


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-9)


def landmark_box(pts, w, h):
    xs, ys = pts[:, 0] * w, pts[:, 1] * h
    return [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "raw" / "hands")
    parser.add_argument("--hand-model", type=Path, default=REPO_ROOT / "models" / "gesture_mlp_hagrid.pt")
    parser.add_argument("--pair-model", type=Path, default=REPO_ROOT / "models" / "pair_mlp.pt")
    parser.add_argument(
        "--every", type=int, default=5, help="use every n-th frame of each gesture (frames are consecutive)"
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "cross_dataset_hands.json")
    args = parser.parse_args()

    hand_model = GestureClassifier.load(args.hand_model)
    pair_model = GestureClassifier.load(args.pair_model)
    records = []  # one per annotated sample (a hand, or a two-hand gesture)
    with HandLandmarker(max_hands=2) as landmarker:
        for subject in args.subjects:
            ann = args.root / f"Subject{subject}.txt"
            archive = args.root / f"Subject{subject}.zip"
            if not (ann.exists() and archive.exists()):
                print(f"Subject {subject}: missing files, skipped")
                continue
            with zipfile.ZipFile(archive) as zf:
                names = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".png")}
                by_frame = defaultdict(list)
                for rgb, col, b in parse_annotations(ann):
                    by_frame[Path(rgb.replace("\\", "/")).name].append((col, b))
                frames = sorted(by_frame, key=lambda n: int(re.match(r"\d+", n).group()))[:: args.every]
                for name in frames:
                    if name not in names:
                        continue
                    img = cv2.imdecode(np.frombuffer(zf.read(names[name]), np.uint8), cv2.IMREAD_COLOR)
                    h, w = img.shape[:2]
                    hands = landmarker.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    anns = by_frame[name]
                    gesture = base_gesture(anns[0][0])
                    if gesture in TWO_HAND_MAP:
                        target = TWO_HAND_MAP[gesture]
                        rec = {
                            "subject": subject,
                            "frame": name,
                            "gesture": gesture,
                            "target": target,
                            "two_hand": True,
                        }
                        if len(hands) >= 2:
                            pts = np.stack([hands[0][0], hands[1][0]])[None]
                            probs = pair_model.predict_proba(pair_features(pts, w / h))[0]
                            rec.update(
                                detected=True, pred=pair_model.classes[int(probs.argmax())], conf=float(probs.max())
                            )
                        else:
                            rec.update(detected=False, pred=None)
                        records.append(rec)
                        continue
                    for col, b in anns:
                        target = ONE_HAND_MAP.get(base_gesture(col))
                        rec = {"subject": subject, "frame": name, "gesture": base_gesture(col), "column": col,
                               "target": target, "two_hand": False}  # fmt: skip
                        best = max(hands, key=lambda hd: iou(landmark_box(hd[0], w, h), b), default=None)
                        if best is None or iou(landmark_box(best[0], w, h), b) < 0.2:
                            rec.update(detected=False, pred=None)
                        else:
                            probs = hand_model.predict_proba(featurize(best[0][None], w / h))[0]
                            rec.update(
                                detected=True, pred=hand_model.classes[int(probs.argmax())], conf=float(probs.max())
                            )
                        records.append(rec)
            print(f"Subject {subject}: {sum(r['subject'] == subject for r in records)} samples")

    report = summarise(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for r in records:
        r["target"] = sorted(r["target"]) if r["target"] else None
    args.out.write_text(json.dumps({"summary": report, "records": records}, indent=1))
    print(json.dumps(report, indent=1))


def summarise(records):
    """Accuracy / precision on gestures with a HaGRID equivalent; prediction distribution for the others."""
    out = {}
    for name, two in [("one_hand", False), ("two_hand", True)]:
        mapped = [r for r in records if r["two_hand"] == two and r["target"]]
        det = [r for r in mapped if r["detected"]]
        entry = {"n": len(mapped), "landmarker_detection_rate": len(det) / max(len(mapped), 1)}
        if det:
            correct = [r["pred"] in r["target"] for r in det]
            entry["accuracy_given_detection"] = float(np.mean(correct))
            entry["end_to_end_accuracy"] = float(np.sum(correct) / len(mapped))
            entry["rejected_as_no_gesture"] = float(np.mean([r["pred"] == "no_gesture" for r in det]))
            # Precision: of the predictions that name a target gesture, how many are right.
            named = [r for r in det if r["pred"] != "no_gesture"]
            entry["precision_of_named_predictions"] = (
                float(np.mean([r["pred"] in r["target"] for r in named])) if named else None
            )
            per = defaultdict(list)
            for r, c in zip(det, correct, strict=True):
                per[r["gesture"]].append(c)
            entry["per_gesture_accuracy"] = {g: round(float(np.mean(v)), 4) for g, v in sorted(per.items())}
            entry["per_gesture_top_predictions"] = {
                g: Counter(r["pred"] for r in det if r["gesture"] == g).most_common(3) for g in sorted(per)
            }
        out[name] = entry
        other = [r for r in records if r["two_hand"] == two and not r["target"] and r["detected"]]
        out[f"{name}_without_equivalent"] = {
            g: Counter(r["pred"] for r in other if r["gesture"] == g).most_common(3)
            for g in sorted({r["gesture"] for r in other})
        }
    return out


if __name__ == "__main__":
    main()
