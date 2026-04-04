"""Dataset loading and preparation utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from guardaikids.config import IMAGE_ANALYSIS_MODEL, IMAGE_FEATURE_DIMS, MODE, TARGET_LABELS, default_image_feature_dir

REQUIRED_SOURCE_COLUMNS = {"video_id", "harm_cat", "title", "description", "transcript"}
THUMBNAIL_LABEL_COLUMN = "thumbnail_harm_cat"


def validate_data_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Dataset path is not a file: {resolved}")
    return resolved


def validate_source_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def load_raw_data(harmful_path: str | Path, harmless_path: str | Path) -> pd.DataFrame:
    harmful_df = pd.read_excel(validate_data_file(harmful_path))
    harmless_df = pd.read_excel(validate_data_file(harmless_path))
    validate_source_columns(harmful_df)
    validate_source_columns(harmless_df)
    return pd.concat([harmful_df, harmless_df], ignore_index=True)


def encode_labels(value: str, target_labels: list[str] | None = None) -> list[str]:
    labels = target_labels or TARGET_LABELS
    categories = [item.strip() for item in str(value).split(",") if item.strip()]
    return [item for item in categories if item in labels]


def _build_missing_image_features(image_feature_dim: int | None = None) -> np.ndarray:
    dim = image_feature_dim or IMAGE_FEATURE_DIMS[IMAGE_ANALYSIS_MODEL]
    missing = np.zeros(dim, dtype=np.float32)
    missing[-1] = 1.0
    return missing


def load_image_features(
    video_id: str,
    image_feature_dir: str | Path | None = None,
    image_feature_dim: int | None = None,
) -> list[float]:
    image_feature_dim = image_feature_dim or IMAGE_FEATURE_DIMS[IMAGE_ANALYSIS_MODEL]
    feature_dir = Path(image_feature_dir) if image_feature_dir else default_image_feature_dir()
    feature_path = feature_dir / f"{video_id}.npy"
    missing = _build_missing_image_features(image_feature_dim)  # dim already resolved above
    if not feature_path.exists():
        return missing.tolist()

    vector = np.load(feature_path)
    if vector.shape != (image_feature_dim,):
        return missing.tolist()
    if not np.isfinite(vector).all():
        return missing.tolist()
    return vector.astype(np.float32).tolist()


def prepare_model_dataframe(
    df: pd.DataFrame,
    target_labels: list[str] | None = None,
    mode: str = MODE,
) -> pd.DataFrame:
    labels = target_labels or TARGET_LABELS
    prepared = df.copy()
    prepared["harm_cat"] = prepared["harm_cat"].fillna("")
    label_series = prepared["harm_cat"]
    label_source = pd.Series("video", index=prepared.index)
    if mode == "image" and THUMBNAIL_LABEL_COLUMN in prepared.columns:
        thumbnail_labels = prepared[THUMBNAIL_LABEL_COLUMN].fillna("").astype(str)
        use_thumbnail_labels = thumbnail_labels.str.strip() != ""
        label_series = label_series.astype(str).copy()
        label_series.loc[use_thumbnail_labels] = thumbnail_labels.loc[use_thumbnail_labels]
        label_source.loc[use_thumbnail_labels] = "thumbnail"

    prepared["label_source"] = label_source
    prepared["filtered_categories"] = label_series.apply(lambda row: encode_labels(row, labels))
    prepared = prepared[
        (label_series.astype(str) == "") | (prepared["filtered_categories"].apply(len) > 0)
    ].copy()

    for label in labels:
        prepared[label] = prepared["filtered_categories"].apply(lambda items: int(label in items))

    prepared["text"] = (
        prepared[["title", "description", "transcript"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )

    model_df = prepared[["video_id", "text", "label_source"] + labels].copy()
    model_df["video_id"] = model_df["video_id"].astype(str)
    model_df["text"] = model_df["text"].astype(str)
    for label in labels:
        model_df[label] = model_df[label].astype(int)
    return model_df


def split_train_validation(
    model_df: pd.DataFrame,
    target_labels: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = target_labels or TARGET_LABELS
    label_counts = model_df[labels].sum(axis=0)
    if (label_counts == 0).any():
        empty_labels = ", ".join(label_counts[label_counts == 0].index.tolist())
        raise ValueError(f"Cannot train because these labels have no positive samples: {empty_labels}")

    train_df, val_df = train_test_split(model_df, test_size=test_size, random_state=random_state)
    val_label_counts = val_df[labels].sum(axis=0)
    if (val_label_counts == 0).any():
        missing_labels = ", ".join(val_label_counts[val_label_counts == 0].index.tolist())
        raise ValueError(
            "Validation split is missing positive samples for labels: "
            f"{missing_labels}. Adjust the split or dataset so evaluation is meaningful."
        )

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["labels"] = train_df[labels].values.tolist()
    val_df["labels"] = val_df[labels].values.tolist()
    train_df["labels"] = train_df["labels"].apply(lambda row: [float(item) for item in row])
    val_df["labels"] = val_df["labels"].apply(lambda row: [float(item) for item in row])
    return train_df, val_df


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df.copy(), preserve_index=False)


def prepare_dataset_inputs(
    dataset: Dataset,
    mode: str = MODE,
    image_feature_dir: str | Path | None = None,
    image_feature_dim: int | None = None,
) -> Dataset:
    if mode not in {"text", "image", "multimodal"}:
        raise ValueError(f"Unsupported mode: {mode}")

    prepared = dataset
    if mode in {"image", "multimodal"}:
        prepared = prepared.map(
            lambda row: {
                "image_features": load_image_features(
                    row["video_id"],
                    image_feature_dir=image_feature_dir,
                    image_feature_dim=image_feature_dim,
                )
            }
        )
    return prepared
