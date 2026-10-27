"""Train the landmark gesture classifier on the HaGRIDv2 cache.

    python -m hgr.train [--classes hagrid|core6] [--negatives all|no_gesture] [--out models/gesture_mlp.pt]

Each epoch draws a class-balanced sample (fresh augmentation every time). After training, the softmax temperature is
fitted on the validation split, and the no_gesture logit bias is tuned there to maximise class-balanced macro F1.
The test split and the personal set are never used for any choice.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hgr.augment import augmented_features
from hgr.features import FEATURE_DIM
from hgr.hagrid import load_cache
from hgr.labels import GESTURE_SETS, HAGRID_CLASSES, REJECT
from hgr.metrics import classification_metrics
from hgr.model import GestureClassifier, GestureMLP

REPO_ROOT = Path(__file__).resolve().parents[2]


def target_label_map(gesture_set="hagrid"):
    """HaGRID class id -> class index in `gesture_set`, including the set's aliases (e.g. peace_inverted -> peace)."""
    spec = GESTURE_SETS[gesture_set]
    classes = spec["classes"]
    names = {c: c for c in classes} | spec["aliases"]
    return {HAGRID_CLASSES.index(h): classes.index(t) for h, t in names.items()}


def selection_mask(arrays, split, negatives, gesture_set="hagrid"):
    """Which hands of `split` are used. `negatives="all"` keeps every hand (other gestures become no_gesture);
    "no_gesture" keeps only the set's classes, their aliases and HaGRID's no_gesture."""
    in_split = arrays["split"] == split
    if negatives == "all":
        return in_split
    if negatives == "no_gesture":
        return in_split & np.isin(arrays["label"], list(target_label_map(gesture_set)))
    raise ValueError(f"negatives must be 'all' or 'no_gesture', got {negatives!r}")


def select(arrays, split, negatives, gesture_set="hagrid"):
    """Landmarks and class indices (within `gesture_set`) for one split."""
    keep = selection_mask(arrays, split, negatives, gesture_set)
    mapping = target_label_map(gesture_set)
    reject = GESTURE_SETS[gesture_set]["classes"].index(REJECT)
    y = np.array([mapping.get(v, reject) for v in arrays["label"][keep]], dtype=np.int64)
    return arrays["landmarks"][keep], y


def balanced_epoch(y, per_class, rng):
    """Indices with `per_class` samples of every class (with replacement only for classes smaller than that)."""
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool, per_class, replace=len(pool) < per_class))
    return rng.permutation(np.concatenate(idx))


def balanced_weights(y, num_classes):
    """Per-sample weights so every class contributes equally (used for validation choices)."""
    counts = np.bincount(y, minlength=num_classes).astype(float)
    return (1.0 / counts)[y]


def predict_logits(model, features, batch=65536):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [model(torch.from_numpy(features[i : i + batch])).numpy() for i in range(0, len(features), batch)]
        )


def fit_temperature(logits, y):
    log_t = torch.zeros(1, requires_grad=True)
    z, t = torch.from_numpy(logits), torch.from_numpy(y)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def balanced_macro_f1(y_true, y_pred, num_classes):
    """Macro F1 as if every class had equal support (re-weights the confusion matrix by 1 / class size)."""
    cm = np.zeros((num_classes, num_classes))
    np.add.at(cm, (y_true, y_pred), balanced_weights(y_true, num_classes))
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(0), 1e-12)
    recall = tp / np.maximum(cm.sum(1), 1e-12)
    return float(np.mean(2 * precision * recall / np.maximum(precision + recall, 1e-12)))


BIAS_GRID = np.linspace(-4, 4, 81)


def tune_reject_bias(logits, y, temperature, reject_index, grid=BIAS_GRID):
    """Bias added to the rejection logit that maximises class-balanced macro F1 (ties -> smallest |bias|)."""
    scores = []
    for b in grid:
        z = logits / temperature
        z[:, reject_index] += b
        scores.append(balanced_macro_f1(y, z.argmax(1), logits.shape[1]))
    scores = np.array(scores)
    best = np.flatnonzero(scores >= scores.max() - 1e-9)
    b = grid[best[np.argmin(np.abs(grid[best]))]]
    return float(b), {f"{g:.1f}": float(s) for g, s in zip(grid, scores, strict=True)}


def add_training_args(parser):
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "processed" / "hagrid_landmarks.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--per-class", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-rotation", type=float, default=20.0, help="rotation augmentation range (degrees)")
    parser.add_argument("--seed", type=int, default=0)


def fit(make_features, y_train, x_val, y_val, classes, args, in_dim=FEATURE_DIM, kind="hand"):
    """Train an MLP, then calibrate it on validation. `make_features(indices, rng)` builds augmented train features.

    Returns the classifier and a training report.
    """
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    num_classes = len(classes)
    model = GestureMLP(num_classes, in_dim=in_dim, hidden=args.hidden, depth=args.depth, dropout=args.dropout)
    model.set_normalization(make_features(rng.choice(len(y_train), min(200_000, len(y_train))), rng))
    steps_per_epoch = num_classes * args.per_class // args.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        idx = balanced_epoch(y_train, args.per_class, rng)
        x = torch.from_numpy(make_features(idx, rng))
        y = torch.from_numpy(y_train[idx])
        model.train()
        total = 0.0
        for step in range(steps_per_epoch):
            sl = slice(step * args.batch_size, (step + 1) * args.batch_size)
            loss = loss_fn(model(x[sl]), y[sl])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        val_f1 = balanced_macro_f1(y_val, predict_logits(model, x_val).argmax(1), num_classes)
        history.append({"epoch": epoch, "loss": total / steps_per_epoch, "val_balanced_macro_f1": val_f1})
        print(
            f"epoch {epoch:2d}  loss {total / steps_per_epoch:.4f}  val bal-F1 {val_f1:.4f}"
            f"  {time.perf_counter() - start:.1f}s",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_logits = predict_logits(model, x_val)
    temperature = fit_temperature(val_logits, y_val)
    reject_bias, bias_curve = tune_reject_bias(val_logits, y_val, temperature, classes.index(REJECT))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {"kind": kind}
    clf = GestureClassifier(model, classes, temperature, reject_bias, REJECT, config=config)
    val_metrics = classification_metrics(y_val, clf.predict(x_val), classes)
    print(f"best val bal-F1 {best_f1:.4f}  temperature {temperature:.3f}  reject bias {reject_bias:+.2f}")
    print(f"val accuracy {val_metrics['accuracy']:.4f}  macro precision {val_metrics['macro_precision']:.4f}")
    report = {"history": history, "temperature": temperature, "reject_bias": reject_bias,
              "bias_curve": bias_curve, "val": val_metrics}  # fmt: skip
    return clf, report


def save(clf, report, out):
    clf.save(out)
    report_path = Path(out).with_suffix(".train.json")
    report_path.write_text(json.dumps(report, indent=1))
    print(f"Saved {out} and {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_training_args(parser)
    parser.add_argument("--classes", choices=sorted(GESTURE_SETS), default="hagrid")
    parser.add_argument("--negatives", choices=["all", "no_gesture"], default="all")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "gesture_mlp.pt")
    args = parser.parse_args()

    classes = GESTURE_SETS[args.classes]["classes"]
    arrays = load_cache(args.cache)
    lm_train, y_train = select(arrays, 0, args.negatives, args.classes)
    lm_val, y_val = select(arrays, 1, args.negatives, args.classes)
    print(f"{len(classes)} classes  train hands {len(y_train):,}  val hands {len(y_val):,}", flush=True)

    # Validation features use aspects drawn from the prior, but fixed (seeded) and without augmentation.
    x_val = augmented_features(lm_val, np.random.default_rng(1), augment=False)

    def make_features(idx, rng):
        return augmented_features(lm_train[idx], rng, max_rotation_deg=args.max_rotation)

    clf, report = fit(make_features, y_train, x_val, y_val, classes, args)
    save(clf, report, args.out)


if __name__ == "__main__":
    main()
