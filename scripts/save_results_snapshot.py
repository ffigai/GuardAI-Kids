"""Save a snapshot of current results and generate comparison graphs for all 7 model configs."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardaikids.config import LABELS_ORDER

# ---------------------------------------------------------------------------
# Model registry — ordered for display
# ---------------------------------------------------------------------------
CONFIGS = [
    ("text only",                       "artifacts/text",                          "predictions_text.json"),
    ("image - clip",                    "artifacts/image",                         "predictions_image.json"),
    ("image - clip_ocr",                "artifacts/image_clip_ocr",                "predictions_image.json"),
    ("image - clip_nsfw_violence",      "artifacts/image_clip_nsfw_violence",      "predictions_image.json"),
    ("multimodal - clip",                "artifacts/multimodal",                    "predictions_multimodal.json"),
    ("multimodal - clip_ocr",           "artifacts/multimodal_clip_ocr",           "predictions_multimodal.json"),
    ("multimodal - clip_nsfw_violence", "artifacts/multimodal_clip_nsfw_violence", "predictions_multimodal.json"),
]

COLORS = [
    "#2ca02c",  # text only          — green
    "#d62728",  # image - clip        — red
    "#ff7f0e",  # image - clip_ocr    — orange
    "#ffcc00",  # image - clip_nsfw   — yellow
    "#1f77b4",  # multimodal - clip   — blue
    "#8c564b",  # multimodal - ocr    — brown
    "#7f7f7f",  # multimodal - nsfw   — grey
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(path: str, pred_file: str):
    root = REPO_ROOT / path
    meta = json.load(open(root / "metadata.json"))
    pred = json.load(open(root / pred_file))
    labels = np.array(pred["labels"])
    probs  = np.array(pred["predictions"])
    lo     = pred.get("label_order") or LABELS_ORDER
    f1t    = meta.get("f1_thresholds") or {}
    preds  = np.zeros_like(probs, dtype=int)
    for i, lbl in enumerate(lo):
        preds[:, i] = (probs[:, i] > f1t.get(lbl, 0.5)).astype(int)
    per_label_f1 = {
        lbl: f1_score(labels[:, i], preds[:, i], zero_division=0)
        for i, lbl in enumerate(lo)
    }
    macro_f1 = sum(per_label_f1.values()) / len(per_label_f1)
    roc_auc  = meta.get("roc_auc") or {}
    mean_auc = sum(roc_auc.values()) / len(roc_auc) if roc_auc else 0.0
    return {
        "meta": meta,
        "labels": labels,
        "probs":  probs,
        "per_label_f1": per_label_f1,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "mean_auc": mean_auc,
    }


def build_summary_df(results: dict) -> pd.DataFrame:
    rows = []
    for name, data in results.items():
        row = {
            "model": name,
            "mean_auc": round(data["mean_auc"], 4),
            "macro_f1": round(data["macro_f1"], 4),
        }
        for lbl in LABELS_ORDER:
            row[f"auc_{lbl}"]  = round(data["roc_auc"].get(lbl, 0), 4)
            row[f"f1_{lbl}"]   = round(data["per_label_f1"].get(lbl, 0), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
def plot_roc_auc_bar(results: dict, output_path: Path) -> None:
    names  = list(results.keys())
    x      = np.arange(len(LABELS_ORDER))
    width  = 0.10
    n      = len(names)
    fig, ax = plt.subplots(figsize=(13, 7))

    for i, (name, data) in enumerate(results.items()):
        vals    = [data["roc_auc"].get(lbl, 0) for lbl in LABELS_ORDER]
        offsets = x + (i - (n - 1) / 2) * width
        ax.bar(offsets, vals, width, label=name, color=COLORS[i],
               edgecolor="black", linewidth=0.5)

    ax.set_title("ROC-AUC by Harm Category — All Models", fontsize=13, pad=12)
    ax.set_xlabel("Harm Category")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_ORDER)
    ax.set_ylim(0.65, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), ncol=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_macro_f1_bar(results: dict, output_path: Path) -> None:
    names  = list(results.keys())
    macro  = [data["macro_f1"] for data in results.values()]
    colors = COLORS[:len(names)]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, macro, color=colors, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, macro):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Macro F1 Score — All Models", fontsize=13, pad=12)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_per_label_f1(results: dict, output_path: Path) -> None:
    names  = list(results.keys())
    x      = np.arange(len(LABELS_ORDER))
    width  = 0.10
    n      = len(names)
    fig, ax = plt.subplots(figsize=(13, 7))

    for i, (name, data) in enumerate(results.items()):
        vals    = [data["per_label_f1"].get(lbl, 0) for lbl in LABELS_ORDER]
        offsets = x + (i - (n - 1) / 2) * width
        ax.bar(offsets, vals, width, label=name, color=COLORS[i],
               edgecolor="black", linewidth=0.5)

    ax.set_title("F1 Score by Harm Category — All Models", fontsize=13, pad=12)
    ax.set_xlabel("Harm Category")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_ORDER)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), ncol=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_roc_curves(results: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for li, lbl in enumerate(LABELS_ORDER):
        ax = axes[li]
        for i, (name, data) in enumerate(results.items()):
            y_true = data["labels"][:, li]
            scores = data["probs"][:, li]
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} ({roc_val:.3f})",
                    color=COLORS[i], linewidth=1.6)
        ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=1)
        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="lower right",
                  bbox_to_anchor=(1.0, 0.0))

    fig.suptitle("ROC Curves by Harm Category — All Models", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_mean_auc_vs_macro_f1(results: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, data) in enumerate(results.items()):
        ax.scatter(data["mean_auc"], data["macro_f1"], s=160,
                   color=COLORS[i], edgecolors="black", linewidth=0.8,
                   zorder=3, label=name)
    ax.set_title("Mean AUC vs Macro F1 — All Models", fontsize=13)
    ax.set_xlabel("Mean ROC-AUC")
    ax.set_ylabel("Macro F1")
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=True, loc="lower right",
              bbox_to_anchor=(1.0, 0.0), borderaxespad=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = REPO_ROOT / "artifacts" / "reports" / f"snapshot_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving snapshot to: {out_dir}\n")

    results = {}
    for name, path, pred_file in CONFIGS:
        try:
            results[name] = load_config(path, pred_file)
            print(f"  Loaded: {name}")
        except FileNotFoundError as e:
            print(f"  SKIP (not found): {name} — {e}")

    if not results:
        print("No results found.")
        return

    # --- numeric snapshot ---
    df = build_summary_df(results)
    csv_path = out_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path.name}")

    raw_snapshot = {
        name: {
            "mean_auc":     data["mean_auc"],
            "macro_f1":     data["macro_f1"],
            "roc_auc":      data["roc_auc"],
            "per_label_f1": data["per_label_f1"],
        }
        for name, data in results.items()
    }
    json_path = out_dir / "results_snapshot.json"
    with open(json_path, "w") as f:
        json.dump(raw_snapshot, f, indent=2)
    print(f"Saved: {json_path.name}")

    # --- graphs ---
    print("\nGenerating graphs...")
    plot_roc_auc_bar(results,           out_dir / "roc_auc_by_label.png")
    plot_macro_f1_bar(results,          out_dir / "macro_f1_all_models.png")
    plot_per_label_f1(results,          out_dir / "f1_by_label.png")
    plot_roc_curves(results,            out_dir / "roc_curves.png")
    plot_mean_auc_vs_macro_f1(results,  out_dir / "auc_vs_f1_scatter.png")

    print(f"\nDone. All files saved to: {out_dir}")


if __name__ == "__main__":
    main()
