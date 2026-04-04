"""Create comparison tables and graphs from text/image/multimodal experiment metadata."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardaikids.config import AGE_GROUPS, LABELS_ORDER, default_artifact_dir

DEFAULT_MODES = ["text", "image", "multimodal"]
MODE_COLORS = {
    "text": "#1f77b4",
    "image": "#ff7f0e",
    "multimodal": "#2ca02c",
}


def ordered_modes(series: pd.Series) -> list[str]:
    available = set(series.dropna().astype(str).tolist())
    return [mode for mode in DEFAULT_MODES if mode in available]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate summary tables and comparison graphs for multimodal experiments."
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=default_artifact_dir(),
        help="Root artifacts directory containing mode subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_artifact_dir() / "reports",
        help="Directory where summary tables and graphs will be written.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=DEFAULT_MODES,
        help="Experiment modes to include. Defaults to text image multimodal.",
    )
    return parser


def load_metadata(artifacts_root: Path, mode: str) -> dict[str, object]:
    metadata_path = artifacts_root / mode / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata for mode '{mode}': {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_predictions(artifacts_root: Path, mode: str) -> dict[str, object]:
    prediction_path = artifacts_root / mode / f"predictions_{mode}.json"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction file for mode '{mode}': {prediction_path}")
    with prediction_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_summary_rows(metadata_by_mode: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overview_rows: list[dict[str, object]] = []
    roc_auc_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []

    for mode, metadata in metadata_by_mode.items():
        overview_rows.append(
            {
                "mode": mode,
                "model_name": metadata.get("model_name"),
                "train_size": metadata.get("train_size"),
                "validation_size": metadata.get("validation_size"),
                "image_feature_dim": metadata.get("image_feature_dim"),
            }
        )
        for label in LABELS_ORDER:
            roc_auc_rows.append(
                {
                    "mode": mode,
                    "label": label,
                    "roc_auc": metadata.get("roc_auc", {}).get(label),
                }
            )
        policy_metrics = metadata.get("policy_metrics", {})
        protection_metrics = metadata.get("protection_metrics", {})
        for age_group in AGE_GROUPS:
            age_metrics = policy_metrics.get(age_group, {})
            policy_rows.append(
                {
                    "mode": mode,
                    "age_group": age_group,
                    "block_precision": age_metrics.get("block_precision"),
                    "block_recall": age_metrics.get("block_recall"),
                    "false_block_rate": age_metrics.get("false_block_rate"),
                    "false_allow_rate": age_metrics.get("false_allow_rate"),
                    "protection_precision": age_metrics.get("protection_precision"),
                    "protection_rate": protection_metrics.get(age_group),
                }
            )

    return (
        pd.DataFrame(overview_rows),
        pd.DataFrame(roc_auc_rows),
        pd.DataFrame(policy_rows),
    )


def save_markdown_tables(overview_df: pd.DataFrame, roc_auc_df: pd.DataFrame, policy_df: pd.DataFrame, output_dir: Path) -> None:
    report_path = output_dir / "summary_report.md"
    roc_auc_pivot = roc_auc_df.pivot(index="label", columns="mode", values="roc_auc")
    policy_pivot = policy_df.pivot_table(
        index="age_group",
        columns="mode",
        values=["block_precision", "block_recall", "protection_rate"],
    )
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Experiment Summary\n\n")
        handle.write("## Overview\n\n")
        handle.write(overview_df.to_markdown(index=False))
        handle.write("\n\n## ROC-AUC By Label\n\n")
        handle.write(roc_auc_pivot.to_markdown())
        handle.write("\n\n## Key Policy Metrics By Age Group\n\n")
        handle.write(policy_pivot.to_markdown())
        handle.write("\n")


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def plot_grouped_bars(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    pivot = df.pivot(index=category_col, columns="mode", values=value_col)
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(category_col.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.legend(title="Mode")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_auc_comparison(roc_auc_df: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        roc_auc_df.pivot(index="label", columns="mode", values="roc_auc")
        .reindex(index=LABELS_ORDER, columns=ordered_modes(roc_auc_df["mode"]))
    )
    modes = list(pivot.columns)
    labels = list(pivot.index)
    x_positions = range(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    for offset_index, mode in enumerate(modes):
        offset = (offset_index - (len(modes) - 1) / 2) * width
        bar_positions = [x + offset for x in x_positions]
        values = pivot[mode].tolist()
        bars = ax.bar(
            bar_positions,
            values,
            width=width,
            label=mode.title(),
            color=MODE_COLORS.get(mode, "#4c4c4c"),
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("ROC-AUC by Harm Category", fontsize=14, pad=12)
    ax.set_xlabel("Harm Category", fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.65, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Mode", frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(predictions_by_mode: dict[str, dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for label_index, label in enumerate(LABELS_ORDER):
        ax = axes[label_index]
        for mode in ordered_modes(pd.Series(list(predictions_by_mode.keys()))):
            payload = predictions_by_mode[mode]
            labels = payload.get("labels", [])
            logits = payload.get("logits")
            predictions = payload.get("predictions")
            if logits is not None:
                scores = [sigmoid(row[label_index]) for row in logits]
            elif predictions is not None:
                scores = [row[label_index] for row in predictions]
            else:
                continue
            y_true = [row[label_index] for row in labels]
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc_score = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                label=f"{mode.title()} (AUC={roc_auc_score:.3f})",
                color=MODE_COLORS.get(mode, "#4c4c4c"),
                linewidth=2,
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("ROC Curves by Harm Category", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_policy_metric_by_age(
    policy_df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    output_path: Path,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> None:
    age_order = list(AGE_GROUPS)
    pivot = (
        policy_df.pivot(index="age_group", columns="mode", values=metric_col)
        .reindex(index=age_order, columns=ordered_modes(policy_df["mode"]))
    )
    modes = list(pivot.columns)
    ages = ["0-4", "5-8", "9-12"]
    x_positions = range(len(age_order))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    for offset_index, mode in enumerate(modes):
        offset = (offset_index - (len(modes) - 1) / 2) * width
        bar_positions = [x + offset for x in x_positions]
        values = pivot[mode].tolist()
        bars = ax.bar(
            bar_positions,
            values,
            width=width,
            label=mode.title(),
            color=MODE_COLORS.get(mode, "#4c4c4c"),
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Age Group", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(ages, fontsize=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Mode", frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_mode = {mode: load_metadata(args.artifacts_root, mode) for mode in args.modes}
    predictions_by_mode = {mode: load_predictions(args.artifacts_root, mode) for mode in args.modes}
    overview_df, roc_auc_df, policy_df = build_summary_rows(metadata_by_mode)

    overview_df.to_csv(output_dir / "overview.csv", index=False)
    roc_auc_df.to_csv(output_dir / "roc_auc_by_label.csv", index=False)
    policy_df.to_csv(output_dir / "policy_metrics_by_age.csv", index=False)
    save_markdown_tables(overview_df, roc_auc_df, policy_df, output_dir)

    plot_roc_auc_comparison(roc_auc_df, output_dir / "roc_auc_by_label.png")
    plot_roc_curves(predictions_by_mode, output_dir / "roc_curves_by_label.png")
    plot_policy_metric_by_age(
        policy_df,
        metric_col="protection_rate",
        title="Protection Rate by Age Group",
        y_label="Protection Rate",
        output_path=output_dir / "protection_rate_by_age.png",
        y_min=0.0,
        y_max=1.05,
    )
    plot_policy_metric_by_age(
        policy_df,
        metric_col="block_precision",
        title="Block Precision by Age Group",
        y_label="Block Precision",
        output_path=output_dir / "block_precision_by_age.png",
        y_min=0.0,
        y_max=1.10,
    )
    plot_policy_metric_by_age(
        policy_df,
        metric_col="block_recall",
        title="Block Recall by Age Group",
        y_label="Block Recall",
        output_path=output_dir / "block_recall_by_age.png",
        y_min=0.0,
        y_max=0.80,
    )

    print(f"Saved report files to: {output_dir}")
    print(f"- {output_dir / 'overview.csv'}")
    print(f"- {output_dir / 'roc_auc_by_label.csv'}")
    print(f"- {output_dir / 'policy_metrics_by_age.csv'}")
    print(f"- {output_dir / 'summary_report.md'}")
    print(f"- {output_dir / 'roc_auc_by_label.png'}")
    print(f"- {output_dir / 'roc_curves_by_label.png'}")
    print(f"- {output_dir / 'protection_rate_by_age.png'}")
    print(f"- {output_dir / 'block_precision_by_age.png'}")
    print(f"- {output_dir / 'block_recall_by_age.png'}")


if __name__ == "__main__":
    main()
