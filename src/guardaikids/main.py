"""CLI entrypoint for training the GuardAI Kids system and analyzing YouTube URLs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from guardaikids.config import AGE_GROUPS, IMAGE_ANALYSIS_MODEL, MODE, XAI_METHOD, default_data_dir
from guardaikids.service import analyze_youtube_url, resolve_artifact_dir, train_and_save_system

VALID_MODES = ("text", "image", "multimodal")
VALID_IMAGE_MODELS = ("clip",)
VALID_XAI_METHODS = ("gradient_tokens",)


def resolve_data_paths() -> tuple[Path, Path]:
    data_dir = default_data_dir()
    harmful = Path(os.environ.get("ETP_HARMFUL_XLSX", data_dir / "Harmful.xlsx"))
    harmless = Path(os.environ.get("ETP_HARMLESS_XLSX", data_dir / "Harmless.xlsx"))
    return harmful, harmless


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the age-aware YouTube safety model or analyze a YouTube URL."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model and save artifacts.")
    train_parser.add_argument("--harmful", type=Path, help="Path to Harmful.xlsx")
    train_parser.add_argument("--harmless", type=Path, help="Path to Harmless.xlsx")
    train_parser.add_argument("--artifact-dir", type=Path, help="Directory to store trained artifacts")
    train_parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        help=f"Override config mode for this training run. Defaults to config MODE ({MODE}).",
    )
    train_parser.add_argument(
        "--image-model",
        choices=VALID_IMAGE_MODELS,
        help=f"Override config image analysis model for this run. Defaults to {IMAGE_ANALYSIS_MODEL}.",
    )
    train_parser.add_argument(
        "--xai-method",
        choices=VALID_XAI_METHODS,
        help=f"Override config XAI method for this run. Defaults to {XAI_METHOD}.",
    )

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a YouTube URL using saved artifacts.")
    analyze_parser.add_argument("--url", required=True, help="YouTube URL to analyze")
    analyze_parser.add_argument("--artifact-dir", type=Path, help="Directory containing trained artifacts")
    analyze_parser.add_argument("--api-key", help="YouTube API key; falls back to YOUTUBE_API_KEY")
    analyze_parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        help=f"Override config mode for this analysis run. Defaults to config MODE ({MODE}).",
    )
    analyze_parser.add_argument(
        "--image-model",
        choices=VALID_IMAGE_MODELS,
        help=f"Override config image analysis model for this run. Defaults to {IMAGE_ANALYSIS_MODEL}.",
    )
    analyze_parser.add_argument(
        "--xai-method",
        choices=VALID_XAI_METHODS,
        help=f"Override config XAI method for this run. Defaults to {XAI_METHOD}.",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full analysis result as JSON instead of text output.",
    )
    return parser


def print_training_summary(results: dict[str, object], artifact_dir: Path) -> None:
    print(f"Artifacts saved to: {artifact_dir}")
    print("Train size:", len(results["train_df"]))
    print("Validation size:", len(results["val_df"]))
    print(results["default_summary"]["classification_report"])
    print("Policy metrics:")
    for age_group, metrics in results["policy_metrics"].items():
        print(f"  {age_group}: {metrics}")


def print_analysis_summary(result: dict[str, object]) -> None:
    metadata = result["metadata"]
    print("Video title:", metadata["title"])
    print("Channel:", metadata["channel"])
    print("Published at:", metadata["published_at"])
    print("Model scores:", result["model_scores"])
    for age_group in AGE_GROUPS:
        recommendation = result["recommendations"][age_group]
        print(f"{age_group}: {recommendation['decision']}")
        print(f"  Explanation: {recommendation['explanation']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        harmful_path, harmless_path = resolve_data_paths()
        if args.harmful:
            harmful_path = args.harmful
        if args.harmless:
            harmless_path = args.harmless
        results, artifact_dir = train_and_save_system(
            harmful_path,
            harmless_path,
            args.artifact_dir,
            mode=args.mode,
            image_analysis_model=args.image_model,
            xai_method=args.xai_method,
        )
        print_training_summary(results, artifact_dir)
        return

    api_key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    result = analyze_youtube_url(
        args.url,
        api_key,
        args.artifact_dir or resolve_artifact_dir(mode=args.mode),
        mode=args.mode,
        image_analysis_model=args.image_model,
        xai_method=args.xai_method,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_analysis_summary(result)


if __name__ == "__main__":
    main()
