"""Shared configuration and constants."""

from pathlib import Path

TARGET_LABELS = ["ADD", "SXL", "PH", "HH"]
LABELS_ORDER = TARGET_LABELS.copy()
AGE_GROUPS = ["0_4", "5_8", "9_12"]
MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 256

DEFAULT_THRESHOLDS = {
    "0_4": {
        "ADD": {"warn": 0.15, "block": 0.95},
        "SXL": {"warn": 0.10, "block": 0.70},
        "PH": {"warn": 0.10, "block": 0.80},
        "HH": {"warn": 0.20, "block": 0.80},
    },
    "5_8": {
        "ADD": {"warn": 0.20, "block": 0.97},
        "SXL": {"warn": 0.15, "block": 0.80},
        "PH": {"warn": 0.15, "block": 0.85},
        "HH": {"warn": 0.20, "block": 0.85},
    },
    "9_12": {
        "ADD": {"warn": 0.25, "block": 0.98},
        "SXL": {"warn": 0.20, "block": 0.85},
        "PH": {"warn": 0.20, "block": 0.90},
        "HH": {"warn": 0.25, "block": 0.90},
    },
}

CATEGORY_DESCRIPTIONS = {
    "ADD": "addictive or substance-related content",
    "SXL": "sexual or explicit material",
    "PH": "physical harm or self-injury references",
    "HH": "hate speech or harassment-related content",
}


def default_data_dir() -> Path:
    return Path.cwd() / "data"


def default_artifact_dir() -> Path:
    return Path.cwd() / "artifacts"
