"""Analyze token-length distribution for the MetaHarm training inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardaikids.config import MODEL_NAME, default_artifact_dir, default_data_dir
from guardaikids.data import load_raw_data, prepare_model_dataframe
from transformers import AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure token-length distribution for the GuardAI Kids training dataset."
    )
    parser.add_argument(
        "--harmful",
        type=Path,
        default=default_data_dir() / "Harmful.xlsx",
        help="Path to Harmful.xlsx",
    )
    parser.add_argument(
        "--harmless",
        type=Path,
        default=default_data_dir() / "Harmless.xlsx",
        help="Path to Harmless.xlsx",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Tokenizer model name to use for counting tokens.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_artifact_dir() / "token_length_report.json",
        help="Where to save the JSON summary report.",
    )
    return parser


def summarize_lengths(lengths: np.ndarray) -> dict[str, object]:
    thresholds = [128, 256, 384, 512]
    percentiles = [50, 75, 90, 95, 99]

    return {
        "sample_count": int(len(lengths)),
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "max": int(lengths.max()),
        "percentiles": {f"p{value}": float(np.percentile(lengths, value)) for value in percentiles},
        "thresholds": {
            str(limit): {
                "samples_over_limit": int((lengths > limit).sum()),
                "share_over_limit": float((lengths > limit).mean()),
            }
            for limit in thresholds
        },
    }


def print_summary(summary: dict[str, object]) -> None:
    print("Token length analysis")
    print("---------------------")
    print("Samples:", summary["sample_count"])
    print("Min:", summary["min"])
    print("Mean:", f"{summary['mean']:.2f}")
    print("Max:", summary["max"])
    print("Percentiles:")
    for label, value in summary["percentiles"].items():
        print(f"  {label}: {value:.0f}")

    print("Truncation impact by max_length:")
    for limit, stats in summary["thresholds"].items():
        share = stats["share_over_limit"] * 100
        print(
            f"  {limit}: {stats['samples_over_limit']} samples "
            f"({share:.2f}%) would be truncated"
        )

    print()
    print("How to read this:")
    print("  - If many samples are over 256, moving to 512 may help.")
    print("  - If very few samples are over 256, 512 is unlikely to help much.")
    print("  - p95 and p99 are especially useful for choosing a practical cutoff.")


def main() -> None:
    args = build_parser().parse_args()

    raw_df = load_raw_data(args.harmful, args.harmless)
    model_df = prepare_model_dataframe(raw_df)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    encoded = tokenizer(
        model_df["text"].tolist(),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    lengths = np.array([len(item) for item in encoded["input_ids"]], dtype=np.int32)
    summary = summarize_lengths(lengths)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print_summary(summary)
    print(f"Saved JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
