"""Classification metrics: accuracy, per-class and macro precision/recall/F1, confusion matrix, calibration error."""

import numpy as np


def confusion(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def classification_metrics(y_true, y_pred, classes, gesture_indices=None):
    """Metrics dict. Macro averages cover `gesture_indices` (default: all classes present in y_true or y_pred)."""
    cm = confusion(y_true, y_pred, len(classes))
    tp = np.diag(cm).astype(float)
    predicted, actual = cm.sum(0), cm.sum(1)
    precision = np.divide(tp, predicted, out=np.full(len(classes), np.nan), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.full(len(classes), np.nan), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.full(len(classes), np.nan),
                   where=(precision + recall) > 0)  # fmt: skip
    if gesture_indices is None:
        gesture_indices = [i for i in range(len(classes)) if actual[i] or predicted[i]]
    gesture_indices = list(gesture_indices)
    return {
        "n": int(cm.sum()),
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_precision": float(np.nanmean(precision[gesture_indices])),
        "macro_recall": float(np.nanmean(recall[gesture_indices])),
        "macro_f1": float(np.nanmean(f1[gesture_indices])),
        "per_class": {
            classes[i]: {
                "precision": _clean(precision[i]),
                "recall": _clean(recall[i]),
                "f1": _clean(f1[i]),
                "support": int(actual[i]),
            }
            for i in range(len(classes))
        },
        "confusion": cm.tolist(),
    }


def expected_calibration_error(probs, y_true, bins=15):
    """ECE of the top-1 confidence."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _clean(x):
    return None if np.isnan(x) else float(x)
