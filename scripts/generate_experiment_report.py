"""Create comparison tables and graphs from text/image/multimodal experiment metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardaikids.config import AGE_GROUPS, LABELS_ORDER, default_artifact_dir

DEFAULT_MODES = ["text", "image", "multimodal"]


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


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_mode = {mode: load_metadata(args.artifacts_root, mode) for mode in args.modes}
    overview_df, roc_auc_df, policy_df = build_summary_rows(metadata_by_mode)

    overview_df.to_csv(output_dir / "overview.csv", index=False)
    roc_auc_df.to_csv(output_dir / "roc_auc_by_label.csv", index=False)
    policy_df.to_csv(output_dir / "policy_metrics_by_age.csv", index=False)
    save_markdown_tables(overview_df, roc_auc_df, policy_df, output_dir)

    plot_grouped_bars(
        roc_auc_df,
        category_col="label",
        value_col="roc_auc",
        title="ROC-AUC By Harm Category",
        y_label="ROC-AUC",
        output_path=output_dir / "roc_auc_by_label.png",
    )
    plot_grouped_bars(
        policy_df,
        category_col="age_group",
        value_col="protection_rate",
        title="Protection Rate By Age Group",
        y_label="Protection Rate",
        output_path=output_dir / "protection_rate_by_age.png",
    )
    plot_grouped_bars(
        policy_df,
        category_col="age_group",
        value_col="block_precision",
        title="Block Precision By Age Group",
        y_label="Block Precision",
        output_path=output_dir / "block_precision_by_age.png",
    )
    plot_grouped_bars(
        policy_df,
        category_col="age_group",
        value_col="block_recall",
        title="Block Recall By Age Group",
        y_label="Block Recall",
        output_path=output_dir / "block_recall_by_age.png",
    )

    print(f"Saved report files to: {output_dir}")
    print(f"- {output_dir / 'overview.csv'}")
    print(f"- {output_dir / 'roc_auc_by_label.csv'}")
    print(f"- {output_dir / 'policy_metrics_by_age.csv'}")
    print(f"- {output_dir / 'summary_report.md'}")
    print(f"- {output_dir / 'roc_auc_by_label.png'}")
    print(f"- {output_dir / 'protection_rate_by_age.png'}")
    print(f"- {output_dir / 'block_precision_by_age.png'}")
    print(f"- {output_dir / 'block_recall_by_age.png'}")


if __name__ == "__main__":
    main()
