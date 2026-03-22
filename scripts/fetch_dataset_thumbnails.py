"""Download YouTube thumbnails for the MetaHarm datasets and export enriched Excel files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardaikids.config import default_data_dir

THUMBNAIL_VARIANTS = [
    "maxresdefault.jpg",
    "sddefault.jpg",
    "hqdefault.jpg",
    "mqdefault.jpg",
    "default.jpg",
]

STATUS_DOWNLOADED = "downloaded"
STATUS_MISSING = "missing"
STATUS_INVALID_VIDEO_ID = "invalid_video_id"
STATUS_REQUEST_FAILED = "request_failed"
THUMBNAIL_LABEL_COLUMN = "thumbnail_harm_cat"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download YouTube thumbnails for Harmful.xlsx and Harmless.xlsx."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Directory containing Harmful.xlsx and Harmless.xlsx.",
    )
    parser.add_argument(
        "--thumbnail-dir",
        type=Path,
        default=default_data_dir() / "thumbnails",
        help="Directory where thumbnail image files will be stored.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds for thumbnail downloads.",
    )
    return parser


def build_thumbnail_candidates(video_id: str) -> list[str]:
    return [f"https://i.ytimg.com/vi/{video_id}/{name}" for name in THUMBNAIL_VARIANTS]


def download_thumbnail(video_id: str, thumbnail_dir: Path, timeout: float) -> dict[str, str]:
    cleaned_video_id = str(video_id).strip()
    if not cleaned_video_id or cleaned_video_id.lower() == "nan":
        return {
            "thumbnail_filename": "",
            "thumbnail_path": "",
            "thumbnail_url": "",
            "thumbnail_status": STATUS_INVALID_VIDEO_ID,
        }

    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    output_path = thumbnail_dir / f"{cleaned_video_id}.jpg"

    # Reuse an existing downloaded thumbnail if present.
    if output_path.exists():
        return {
            "thumbnail_filename": output_path.name,
            "thumbnail_path": str(output_path.resolve()),
            "thumbnail_url": build_thumbnail_candidates(cleaned_video_id)[0],
            "thumbnail_status": STATUS_DOWNLOADED,
        }

    last_http_status = None
    for url in build_thumbnail_candidates(cleaned_video_id):
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    last_http_status = "non_image_response"
                    continue
                output_path.write_bytes(response.read())
                return {
                    "thumbnail_filename": output_path.name,
                    "thumbnail_path": str(output_path.resolve()),
                    "thumbnail_url": url,
                    "thumbnail_status": STATUS_DOWNLOADED,
                }
        except HTTPError as exc:
            if exc.code == 404:
                last_http_status = "404"
                continue
            return {
                "thumbnail_filename": "",
                "thumbnail_path": "",
                "thumbnail_url": url,
                "thumbnail_status": STATUS_REQUEST_FAILED,
            }
        except URLError:
            return {
                "thumbnail_filename": "",
                "thumbnail_path": "",
                "thumbnail_url": url,
                "thumbnail_status": STATUS_REQUEST_FAILED,
            }

    return {
        "thumbnail_filename": "",
        "thumbnail_path": "",
        "thumbnail_url": build_thumbnail_candidates(cleaned_video_id)[-1] if cleaned_video_id else "",
        "thumbnail_status": STATUS_MISSING if last_http_status in {"404", "non_image_response", None} else STATUS_REQUEST_FAILED,
    }


def enrich_dataset(input_path: Path, output_path: Path, thumbnail_dir: Path, timeout: float) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    if "video_id" not in df.columns:
        raise ValueError(f"{input_path} does not contain a 'video_id' column.")
    if "harm_cat" not in df.columns:
        raise ValueError(f"{input_path} does not contain a 'harm_cat' column.")

    thumbnail_info = [
        download_thumbnail(video_id, thumbnail_dir=thumbnail_dir, timeout=timeout)
        for video_id in df["video_id"].tolist()
    ]
    info_df = pd.DataFrame(thumbnail_info)
    enriched = pd.concat([df, info_df], axis=1)
    enriched[THUMBNAIL_LABEL_COLUMN] = df["harm_cat"].fillna("").astype(str)
    enriched.to_excel(output_path, index=False)
    return enriched


def summarize_statuses(name: str, enriched_df: pd.DataFrame) -> None:
    print(f"{name}:")
    status_counts = enriched_df["thumbnail_status"].value_counts(dropna=False).to_dict()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir
    thumbnail_dir = args.thumbnail_dir

    jobs = [
        (data_dir / "Harmful.xlsx", data_dir / "Harmful_with_thumbnails.xlsx"),
        (data_dir / "Harmless.xlsx", data_dir / "Harmless_with_thumbnails.xlsx"),
    ]

    for input_path, output_path in jobs:
        print(f"Processing {input_path.name}...")
        enriched = enrich_dataset(
            input_path=input_path,
            output_path=output_path,
            thumbnail_dir=thumbnail_dir,
            timeout=args.timeout,
        )
        summarize_statuses(input_path.name, enriched)
        print(f"  wrote: {output_path}")


if __name__ == "__main__":
    main()
