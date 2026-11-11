"""Render README visuals: a demo GIF from the real pipeline and result charts.

Usage: python scripts/make_figures.py
Writes to docs/figures/.
"""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hgr.model import GestureClassifier
from hgr.pipeline import HAND_CONNECTIONS, GesturePipeline

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
BG, FG, MUTED, ACCENT, ACCENT2 = "#0d1117", "#e6edf3", "#8b949e", "#58e6c1", "#ff7a59"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#30363d", "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def demo_gif(src=ROOT / "web" / "test-media" / "demo.webm", out=FIG / "demo.gif", width=480, step=2):
    clf = GestureClassifier.load(ROOT / "models" / "gesture_mlp_hagrid.pt")
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, i = [], 0
    with GesturePipeline(clf) as pipe:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            preds = pipe.process(frame, int(i * 1000 / fps))
            i += 1
            if i % step:
                continue
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (width, int(h * width / w)))
            h, w = frame.shape[:2]
            overlay = (frame * 0.55).astype(np.uint8)
            for p in preds:
                pts = (p.landmarks * [w, h]).astype(int)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), (193, 230, 88), 2, cv2.LINE_AA)
                for q in pts:
                    cv2.circle(overlay, tuple(q), 3, (255, 255, 255), -1, cv2.LINE_AA)
                x, y = pts[:, 0].min(), max(pts[:, 1].min() - 12, 18)
                txt = f"{p.label}  {p.confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                cv2.rectangle(overlay, (x - 4, y - th - 6), (x + tw + 4, y + 4), (89, 122, 255), -1)
                cv2.putText(overlay, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)))
    cap.release()
    frames = frames[:150]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * step / fps), loop=0, optimize=True)
    print("wrote", out, len(frames), "frames")


def results():
    m = json.load(open(ROOT / "reports" / "metrics_hagrid.json"))
    core = m["hagrid_test_core"]
    classes = m["classes"]

    # Confusion matrix (row-normalised)
    cm = np.array(core["confusion"], float)
    cm = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"HaGRIDv2 test confusion  ·  {core['n']:,} hands  ·  acc {core['accuracy']:.1%}", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "confusion.png", dpi=130)
    plt.close(fig)

    # Per-class F1
    pc = core["per_class"]
    items = sorted(((k, v["f1"]) for k, v in pc.items() if isinstance(v, dict) and "f1" in v), key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(9, 6))
    names, f1 = zip(*items)
    ax.barh(names, f1, color=[ACCENT2 if v < 0.97 else ACCENT for v in f1])
    ax.set_xlim(min(f1) - 0.02, 1.0)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Per-gesture F1 on held-out subjects", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "per_class_f1.png", dpi=130)
    plt.close(fig)

    # Headline metrics + fairness slices
    sl = m["hagrid_test_core_slices"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    head = {"Accuracy": core["accuracy"], "Macro F1": core["macro_f1"],
            "Open-set acc": m["hagrid_test_open_set"]["accuracy"], "1 − ECE": 1 - core["ece"]}
    axes[0].bar(head.keys(), head.values(), color=ACCENT)
    for i, v in enumerate(head.values()):
        axes[0].text(i, v + 0.002, f"{v:.1%}", ha="center", fontsize=9)
    axes[0].set_ylim(0.9, 1.0)
    axes[0].set_title("Headline metrics", loc="left")
    axes[0].tick_params(axis="x", labelsize=8)
    keys = list(sl)
    vals = [sl[k]["gesture_recall"] for k in keys]
    axes[1].bar(keys, vals, color="#7aa2ff")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.001, f"{v:.1%}", ha="center", fontsize=8)
    axes[1].set_ylim(min(vals) - 0.01, 1.0)
    axes[1].set_title("Gesture recall across demographic slices", loc="left")
    axes[1].tick_params(axis="x", labelsize=8, rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "headline.png", dpi=130)
    plt.close(fig)

    # Training curves
    h = json.load(open(ROOT / "reports" / "train_history_hagrid.json"))["history"]
    if isinstance(h, list) and h and isinstance(h[0], dict):
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for key, c in (("loss", ACCENT2),):
            if key in h[0]:
                ax.plot([r[key] for r in h], color=c, lw=2, label=key.replace("_", " "))
        accs = [k for k in h[0] if "f1" in k]
        if accs:
            ax2 = ax.twinx()
            ax2.plot([r[accs[0]] for r in h], color="#7aa2ff", lw=2, ls="--", label=accs[0].replace("_", " "))
            ax2.legend(loc="center right", frameon=False)
        ax.set_xlabel("epoch")
        ax.legend(loc="upper right", frameon=False)
        ax.set_title("Training", loc="left")
        fig.tight_layout()
        fig.savefig(FIG / "training.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    results()
    demo_gif()
