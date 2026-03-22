"""Regenerate explanation outputs from saved artifacts without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardaikids.config import AGE_GROUPS, MODE, default_artifact_dir
from guardaikids.explainability import explain_video
from guardaikids.modeling import load_saved_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate validation-set explanations from saved model artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=default_artifact_dir() / MODE,
        help="Artifact directory containing model/, tokenizer/, and predictions_<mode>.json.",
    )
    parser.add_argument(
        "--mode",
        default=MODE,
        help="Mode name used in the predictions file name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of validation rows to explain. 0 means all rows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = args.artifact_dir
    predictions_path = artifact_dir / f"predictions_{args.mode}.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction artifact not found: {predictions_path}")

    with predictions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    model, tokenizer = load_saved_model(artifact_dir / "model", artifact_dir / "tokenizer")
    texts = payload.get("texts", [])
    probabilities = payload.get("predictions", [])
    video_ids = payload.get("video_ids", [])
    row_count = len(probabilities) if args.limit <= 0 else min(args.limit, len(probabilities))

    explanations: list[dict[str, object]] = []
    for index in range(row_count):
        text = texts[index] if index < len(texts) else ""
        probs_row = probabilities[index]
        explanation_by_age = {}
        for age_group in AGE_GROUPS:
            explanation = explain_video(text, probs_row, age_group, model, tokenizer)
            explanation_by_age[age_group] = {
                "decision": explanation["decision"],
                "trigger_category": explanation["trigger_category"],
                "probability": explanation["probability"],
                "threshold": explanation["threshold"],
                "explanation": explanation["human_readable_explanation"],
                "top_tokens": [
                    {"token": token, "importance": float(score)} for token, score in explanation["top_tokens"]
                ],
            }
        explanations.append(
            {
                "index": index,
                "video_id": video_ids[index] if index < len(video_ids) else "",
                "text": text,
                "recommendations": explanation_by_age,
            }
        )

    output_path = artifact_dir / f"explanations_{args.mode}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(explanations, handle, indent=2)

    print(f"Saved explanations to: {output_path}")


if __name__ == "__main__":
    main()
