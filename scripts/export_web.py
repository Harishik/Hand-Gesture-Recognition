"""Export trained classifiers for the browser app: BatchNorm folded into the linear layers, weights as base64 float32.

python scripts/export_web.py [--hand models/gesture_mlp_hagrid.pt] [--pair models/pair_mlp.pt] [--out web/models]

Also writes web/models/parity.json: sample inputs with the Python features and probabilities, which
tests/test_web_parity.py uses to check that the JavaScript port computes exactly the same thing.
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hgr.features import featurize
from hgr.hagrid import load_cache
from hgr.model import GestureClassifier
from hgr.pairs import build_pairs, pair_features

REPO_ROOT = Path(__file__).resolve().parents[1]


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def fold_layers(model):
    """[(W (out, in), b (out,), activation)] with BatchNorm (eval mode) folded into the preceding Linear."""
    layers, modules = [], list(model.net)
    i = 0
    while i < len(modules):
        linear = modules[i]
        assert isinstance(linear, nn.Linear), modules[i]
        w = linear.weight.detach().double().numpy()
        b = linear.bias.detach().double().numpy()
        act = None
        if i + 1 < len(modules) and isinstance(modules[i + 1], nn.BatchNorm1d):
            bn = modules[i + 1]
            s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double().numpy()
            w = w * s[:, None]
            b = (b - bn.running_mean.detach().double().numpy()) * s + bn.bias.detach().double().numpy()
            act = "gelu"
            i += 4  # Linear, BatchNorm, GELU, Dropout
        else:
            i += 1
        layers.append((w, b, act))
    return layers


def export(clf, path):
    model = clf.model.eval()
    payload = {
        "kind": clf.config.get("kind", "hand"),
        "classes": list(clf.classes),
        "reject_index": clf.reject_index,
        "temperature": clf.temperature,
        "reject_bias": clf.reject_bias,
        "in_dim": int(model.mean.numel()),
        "mean": b64(model.mean.numpy()),
        "std": b64(model.std.numpy()),
        "layers": [
            {"out": w.shape[0], "in": w.shape[1], "w": b64(w), "b": b64(b), "act": act}
            for w, b, act in fold_layers(model)
        ],
    }
    path.write_text(json.dumps(payload))
    print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hand", type=Path, default=REPO_ROOT / "models" / "gesture_mlp_hagrid.pt")
    parser.add_argument("--pair", type=Path, default=REPO_ROOT / "models" / "pair_mlp.pt")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "processed" / "hagrid_landmarks.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "web" / "models")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    hand, pair = GestureClassifier.load(args.hand), GestureClassifier.load(args.pair)
    export(hand, args.out / "hand.json")
    export(pair, args.out / "pair.json")

    # Parity fixture: a few real test hands / pairs, with Python's features and probabilities.
    arrays = load_cache(args.cache)
    rng = np.random.default_rng(0)
    test = np.flatnonzero(arrays["split"] == 2)
    hands = arrays["landmarks"][rng.choice(test, 8, replace=False)]
    aspects = np.array([0.75, 1.0, 16 / 9, 9 / 16, 4 / 3, 0.75, 1.0, 1.5], dtype=np.float32)
    pairs, _ = build_pairs(arrays, 2)
    pairs = pairs[rng.choice(len(pairs), 4, replace=False)]
    pair_aspects = aspects[:4]
    hand_feats = np.concatenate([featurize(h[None], a) for h, a in zip(hands, aspects, strict=True)])
    pair_feats = np.concatenate([pair_features(p[None], a) for p, a in zip(pairs, pair_aspects, strict=True)])
    fixture = {
        "hands": [
            {"landmarks": h.tolist(), "aspect": float(a), "features": f.tolist(), "probs": p.tolist()}
            for h, a, f, p in zip(hands, aspects, hand_feats, hand.predict_proba(hand_feats), strict=True)
        ],
        "pairs": [
            {"landmarks": q.tolist(), "aspect": float(a), "features": f.tolist(), "probs": p.tolist()}
            for q, a, f, p in zip(pairs, pair_aspects, pair_feats, pair.predict_proba(pair_feats), strict=True)
        ],
    }
    (args.out / "parity.json").write_text(json.dumps(fixture))
    print(f"Saved {args.out / 'parity.json'}")


if __name__ == "__main__":
    main()
