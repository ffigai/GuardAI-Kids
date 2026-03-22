"""Copy `harm_cat` into `thumbnail_harm_cat` for existing *_with_thumbnails workbooks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardaikids.config import default_data_dir

THUMBNAIL_LABEL_COLUMN = "thumbnail_harm_cat"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill thumbnail_harm_cat into existing *_with_thumbnails Excel files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Directory containing source and *_with_thumbnails Excel files.",
    )
    return parser


def apply_thumbnail_labels(source_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
    if "harm_cat" not in source_df.columns:
        raise ValueError("Source workbook is missing 'harm_cat'.")
    if len(source_df) != len(enriched_df):
        raise ValueError("Source and enriched workbooks do not have the same number of rows.")

    if "video_id" in source_df.columns and "video_id" in enriched_df.columns:
        source_ids = source_df["video_id"].fillna("").astype(str).tolist()
        enriched_ids = enriched_df["video_id"].fillna("").astype(str).tolist()
        if source_ids != enriched_ids:
            raise ValueError("Source and enriched workbooks do not have matching video_id order.")

    updated = enriched_df.copy()
    updated[THUMBNAIL_LABEL_COLUMN] = source_df["harm_cat"].fillna("").astype(str).tolist()
    return updated


def process_pair(source_path: Path, enriched_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Source workbook not found: {source_path}")
    if not enriched_path.exists():
        raise FileNotFoundError(f"Enriched workbook not found: {enriched_path}")

    source_df = pd.read_excel(source_path)
    enriched_df = pd.read_excel(enriched_path)
    updated = apply_thumbnail_labels(source_df, enriched_df)
    updated.to_excel(enriched_path, index=False)
    print(f"Updated {enriched_path.name} with {THUMBNAIL_LABEL_COLUMN}")


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir
    jobs = [
        (data_dir / "Harmful.xlsx", data_dir / "Harmful_with_thumbnails.xlsx"),
        (data_dir / "Harmless.xlsx", data_dir / "Harmless_with_thumbnails.xlsx"),
    ]

    for source_path, enriched_path in jobs:
        process_pair(source_path, enriched_path)


if __name__ == "__main__":
    main()
