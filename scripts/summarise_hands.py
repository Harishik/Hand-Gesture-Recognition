"""Merge per-subject HANDS reports into one summary and chart.

python scripts/summarise_hands.py   # reads reports/cross_dataset_hands_subject*.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_hands import summarise  # noqa: E402

records, per_subject = [], {}
for f in sorted((ROOT / "reports").glob("cross_dataset_hands_subject*.json")):
    d = json.loads(f.read_text())
    records += d["records"]
    s = d["summary"]["one_hand"]
    per_subject[f.stem.rsplit("subject", 1)[1]] = {
        k: s[k] for k in ("n", "landmarker_detection_rate", "accuracy_given_detection", "end_to_end_accuracy")
    }
summary = summarise(records)
summary["per_subject_one_hand"] = per_subject
(ROOT / "reports" / "cross_dataset_hands.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "one_hand_without_equivalent"}, indent=1)[:4000])

# Chart: per-gesture accuracy (given detection) across all subjects
BG, FG, MUTED = "#0d1117", "#e6edf3", "#8b949e"
plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#30363d",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
pg = summary["one_hand"]["per_gesture_accuracy"]
items = sorted(pg.items(), key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(8, 3.6))
names = [k for k, _ in items]
vals = [v for _, v in items]
ax.barh(names, vals, color=["#ff7a59" if v < 0.8 else "#58e6c1" for v in vals])
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
ax.set_xlim(0, 1.08)
oh = summary["one_hand"]
ax.set_title(
    f"Unseen dataset (HANDS, 5 subjects, {oh['n']:,} hands): {oh['accuracy_given_detection']:.1%} when detected",
    loc="left",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "figures" / "cross_dataset.png", dpi=130)
