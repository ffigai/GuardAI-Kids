"""Extract CLIP-based thumbnail features into .npy vectors for multimodal training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardaikids.config import (
    IMAGE_ANALYSIS_MODEL,
    IMAGE_FEATURE_DIMS,
    IMAGE_QUALITY_FEATURE_NAMES,
    IMAGE_SIMILARITY_PROMPTS,
    default_image_feature_dir,
    default_image_feature_dir_for_model,
    default_thumbnail_dir,
)
from guardaikids.image_features import (
    CLIP_MODEL_NAME,
    NSFW_MODEL_NAME,
    VIOLENCE_MODEL_NAME,
    build_feature_vector_from_image,
    build_feature_vector_from_image_with_ocr,
    build_feature_vector_from_image_with_specialists,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CLIP image features from downloaded YouTube thumbnails."
    )
    parser.add_argument(
        "--thumbnail-dir",
        type=Path,
        default=default_thumbnail_dir(),
        help="Directory containing thumbnail .jpg files named by video_id.",
    )
    parser.add_argument(
        "--image-analysis-model",
        choices=["clip", "clip_ocr", "clip_nsfw_violence"],
        default=IMAGE_ANALYSIS_MODEL,
        help=f"Image analysis pipeline to use. Defaults to config IMAGE_ANALYSIS_MODEL ({IMAGE_ANALYSIS_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where .npy image feature files will be written. Defaults to the standard dir for the selected model.",
    )
    parser.add_argument(
        "--model-name",
        default=CLIP_MODEL_NAME,
        help="CLIP model to use for image and text embeddings.",
    )
    return parser


def list_thumbnail_files(thumbnail_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(thumbnail_dir.glob(pattern)))
    return files

def main() -> None:
    args = build_parser().parse_args()
    selected_model = args.image_analysis_model
    thumbnail_dir = args.thumbnail_dir
    output_dir = args.output_dir or default_image_feature_dir_for_model(selected_model)

    if not thumbnail_dir.exists():
        raise FileNotFoundError(f"Thumbnail directory not found: {thumbnail_dir}")

    image_files = list_thumbnail_files(thumbnail_dir)
    if not image_files:
        raise FileNotFoundError(f"No thumbnail image files found in {thumbnail_dir}")

    image_feature_dim = IMAGE_FEATURE_DIMS[selected_model]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "clip_model_name": args.model_name,
        "image_analysis_model": selected_model,
        "feature_dim": image_feature_dim,
        "embedding_dim": 512,
        "similarity_prompt_order": IMAGE_SIMILARITY_PROMPTS,
        "quality_feature_order": IMAGE_QUALITY_FEATURE_NAMES,
        "missing_flag_index": image_feature_dim - 1,
    }
    if selected_model == "clip_nsfw_violence":
        manifest["nsfw_model_name"] = NSFW_MODEL_NAME
        manifest["violence_model_name"] = VIOLENCE_MODEL_NAME
    with (output_dir / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Found {len(image_files)} thumbnail files")
    print(f"Image analysis model: {selected_model} (feature_dim={image_feature_dim})")
    print(f"Writing image features to: {output_dir}")
    for index, image_path in enumerate(image_files, start=1):
        video_id = image_path.stem
        output_path = output_dir / f"{video_id}.npy"
        image = Image.open(image_path).convert("RGB")
        if selected_model == "clip_nsfw_violence":
            vector = build_feature_vector_from_image_with_specialists(image, clip_model_name=args.model_name)
        elif selected_model == "clip_ocr":
            vector = build_feature_vector_from_image_with_ocr(image, model_name=args.model_name)
        else:
            vector = build_feature_vector_from_image(image, model_name=args.model_name)
        np.save(output_path, vector)
        if index % 100 == 0 or index == len(image_files):
            print(f"Processed {index}/{len(image_files)}")

    print("Done.")
    print(f"Feature manifest: {output_dir / 'feature_manifest.json'}")


if __name__ == "__main__":
    main()
