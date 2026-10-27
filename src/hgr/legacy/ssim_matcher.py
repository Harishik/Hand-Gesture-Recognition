"""Original (2024) baseline: SSIM template matching against hand-captured reference photos.

Kept for comparison with the learned models. Run with:

    python -m hgr.legacy.ssim_matcher [--data data/personal]
"""

import argparse
from pathlib import Path

import cv2
from skimage.metrics import structural_similarity as ssim

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "personal"
TEMPLATE_SIZE = (50, 50)
CROP_PADDING = 20  # px around the landmark bbox, roughly matching how the reference photos were framed

GESTURE_DIRS = {
    "Hello": "hello",
    "Fist": "fist",
    "Peace": "peace",
    "Okay": "okay",
    "Thumbs Up": "thumbs_up",
    "Thumbs Down": "thumbs_down",
}


def preprocess(image, is_gray=False):
    gray = image if is_gray else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    return resized / 255.0


def load_templates(data_dir):
    data_dir = Path(data_dir)
    templates = {}
    for gesture, folder in GESTURE_DIRS.items():
        gesture_dir = data_dir / folder
        if not gesture_dir.is_dir():
            raise FileNotFoundError(f"Missing gesture folder: {gesture_dir}")
        images = []
        for path in sorted(gesture_dir.iterdir()):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(preprocess(img, is_gray=True))
        templates[gesture] = images
    return templates


def crop_hand(frame, bbox, padding=CROP_PADDING):
    """Crop a padded bbox (x, y, w, h), clamped to the frame. Returns None if empty."""
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = bbox
    x_min, y_min = max(0, x - padding), max(0, y - padding)
    x_max, y_max = min(frame_w, x + w + padding), min(frame_h, y + h + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    return frame[y_min:y_max, x_min:x_max]


def recognize_gesture(processed_crop, templates):
    """Return (best gesture, best SSIM score in [-1, 1]). Note: a similarity score, not an accuracy."""
    best_score = -1.0
    best_gesture = None
    for gesture, images in templates.items():
        for img in images:
            score = ssim(processed_crop, img, data_range=1.0)
            if score > best_score:
                best_score = score
                best_gesture = gesture
    return best_gesture, best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help="folder with one sub-folder per gesture")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    # Imported here so the helpers above stay importable without the legacy cvzone/mediapipe pin.
    from cvzone.HandTrackingModule import HandDetector

    templates = load_templates(args.data)
    detector = HandDetector(maxHands=2, detectionCon=0.7, minTrackCon=0.7)
    cap = cv2.VideoCapture(args.camera)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # findHands draws on the image it is given, so draw on a copy and crop from the clean frame.
        display = frame.copy()
        hands, display = detector.findHands(display)

        for hand in hands:
            crop = crop_hand(frame, hand["bbox"])
            if crop is None:
                continue
            gesture, score = recognize_gesture(preprocess(crop), templates)
            x, y = hand["bbox"][:2]
            text = f"{gesture} (SSIM {score:.2f})"
            cv2.putText(display, text, (x, max(30, y - 55)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        frame_h, frame_w = display.shape[:2]
        enlarged = cv2.resize(display, (int(frame_w * 1.5), int(frame_h * 1.5)))
        cv2.imshow("Hand Gesture Recognition (legacy SSIM)", enlarged)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
