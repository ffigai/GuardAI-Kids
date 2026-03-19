"""Dataset loading and preparation utilities."""

from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from etp.config import TARGET_LABELS

REQUIRED_SOURCE_COLUMNS = {"harm_cat", "title", "description", "transcript"}


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


def prepare_model_dataframe(df: pd.DataFrame, target_labels: list[str] | None = None) -> pd.DataFrame:
    labels = target_labels or TARGET_LABELS
    prepared = df.copy()
    prepared["harm_cat"] = prepared["harm_cat"].fillna("")
    prepared["filtered_categories"] = prepared["harm_cat"].apply(lambda row: encode_labels(row, labels))
    prepared = prepared[
        (prepared["harm_cat"] == "") | (prepared["filtered_categories"].apply(len) > 0)
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

    model_df = prepared[["text"] + labels].copy()
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
    return Dataset.from_pandas(df[["text", "labels"]].copy(), preserve_index=False)
