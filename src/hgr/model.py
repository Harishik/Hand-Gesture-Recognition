"""Landmark-feature MLP and its checkpoint (weights + standardisation + calibration + rejection bias)."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from hgr.features import FEATURE_DIM


class GestureMLP(nn.Module):
    def __init__(self, num_classes, in_dim=FEATURE_DIM, hidden=256, depth=3, dropout=0.1):
        super().__init__()
        self.arch = {"num_classes": num_classes, "in_dim": in_dim, "hidden": hidden, "depth": depth, "dropout": dropout}
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        layers, dim = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def set_normalization(self, features):
        features = torch.as_tensor(features, dtype=torch.float32)
        self.mean.copy_(features.mean(0))
        self.std.copy_(features.std(0).clamp_min(1e-6))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class GestureClassifier:
    """Inference wrapper: calibrated probabilities with a tunable bias on the rejection class."""

    def __init__(self, model, classes, temperature=1.0, reject_bias=0.0, reject_class="no_gesture", config=None):
        self.model = model.eval()
        self.classes = tuple(classes)
        self.temperature = float(temperature)
        self.reject_bias = float(reject_bias)
        self.reject_index = self.classes.index(reject_class)
        self.config = config or {}

    @torch.no_grad()
    def logits(self, features):
        return self.model(torch.as_tensor(features, dtype=torch.float32)).numpy()

    def adjust(self, logits):
        """Apply temperature and rejection bias to raw logits."""
        z = logits / self.temperature
        z[:, self.reject_index] += self.reject_bias
        return z

    def predict_proba(self, features):
        z = self.adjust(self.logits(features))
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, features):
        return self.predict_proba(features).argmax(axis=1)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": self.model.arch,
                "state_dict": self.model.state_dict(),
                "classes": self.classes,
                "temperature": self.temperature,
                "reject_bias": self.reject_bias,
                "reject_class": self.classes[self.reject_index],
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        arch = ckpt.get("arch") or {  # checkpoints saved before "arch" was stored
            "num_classes": len(ckpt["classes"]),
            **{k: ckpt["config"][k] for k in ("hidden", "depth", "dropout")},
        }
        model = GestureMLP(**arch)
        model.load_state_dict(ckpt["state_dict"])
        return cls(
            model,
            ckpt["classes"],
            ckpt["temperature"],
            ckpt["reject_bias"],
            ckpt["reject_class"],
            ckpt["config"],
        )
