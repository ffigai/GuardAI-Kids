"""Shared configuration and constants."""

from pathlib import Path

TARGET_LABELS = ["ADD", "SXL", "PH", "HH"]
LABELS_ORDER = TARGET_LABELS.copy()
MODEL_NAME = "distilroberta-base"
MODE = "multimodal" # "text", "image", or "multimodal"
MAX_LENGTH = 512
IMAGE_ANALYSIS_MODEL = "clip_nsfw_violence"  # "clip", "clip_ocr", or "clip_nsfw_violence"
XAI_METHOD = "gradient_tokens"   # later: "integrated_gradients", "occlusion"

IMAGE_SIMILARITY_PROMPTS = [
    "a child-friendly harmless youtube thumbnail",
    "a youtube thumbnail showing addictive or substance-related content",
    "a youtube thumbnail showing sexual or explicit content",
    "a youtube thumbnail showing physical harm or self-injury",
    "a youtube thumbnail showing hate speech or harassment",
    "a youtube thumbnail showing weapons, threats, or intimidation",
    "a youtube thumbnail designed to be disturbing or scary for children",
    "a youtube thumbnail focused on risky stunts, pranks, or dangerous challenges",
]
IMAGE_QUALITY_FEATURE_NAMES = [
    "brightness_mean",
    "contrast_std",
    "saturation_mean",
    "colorfulness",
    "detail_score",
]
_CLIP_BASE_DIM = 512 + len(IMAGE_SIMILARITY_PROMPTS) + len(IMAGE_QUALITY_FEATURE_NAMES) + 1
# clip_nsfw_violence layout (after quality features, before missing flag):
#   +1 NSFW score (Marqo/nsfw-image-detection-384)
#   +1 violence score (jaranohaal/vit-base-violence-detection)
#   +1 has_ocr_text flag (gate: 1.0 if text found, else 0.0)
#   +N OCR harm similarity scores (same N as IMAGE_SIMILARITY_PROMPTS, zeroed when flag=0)
_CLIP_SPECIALIST_DIM = _CLIP_BASE_DIM + 2 + 1 + len(IMAGE_SIMILARITY_PROMPTS)
IMAGE_FEATURE_DIMS = {
    "clip":                 _CLIP_BASE_DIM,
    "clip_ocr":             _CLIP_BASE_DIM + len(IMAGE_SIMILARITY_PROMPTS) + 1,  # legacy: OCR prompt scores + has_ocr_text flag
    "clip_nsfw_violence":   _CLIP_SPECIALIST_DIM,
}
IMAGE_FEATURE_DIM = IMAGE_FEATURE_DIMS[IMAGE_ANALYSIS_MODEL]

# Default f2 thresholds per mode — recall-weighted, loaded from artifact metadata at runtime.
# These are fallbacks used when no artifact is loaded (e.g. tests, CLI without --artifact-dir).
DEFAULT_F2_THRESHOLDS = {
    "text":       {"ADD": 0.25, "SXL": 0.45, "PH": 0.30, "HH": 0.15},
    "image":      {"ADD": 0.30, "SXL": 0.30, "PH": 0.30, "HH": 0.20},
    "multimodal": {"ADD": 0.40, "SXL": 0.30, "PH": 0.20, "HH": 0.20},
}

CATEGORY_DESCRIPTIONS = {
    "ADD": "addictive or substance-related content",
    "SXL": "sexual or explicit material",
    "PH": "physical harm or self-injury references",
    "HH": "hate speech or harassment-related content",
}


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def default_image_feature_dir() -> Path:
    return default_data_dir() / "image_features"


def default_image_feature_dir_for_model(model_name: str | None = None) -> Path:
    name = model_name or IMAGE_ANALYSIS_MODEL
    if name == "clip":
        return default_image_feature_dir()
    return default_data_dir() / f"image_features_{name}"


def default_thumbnail_dir() -> Path:
    return default_data_dir() / "thumbnails"


def default_artifact_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts"


def get_default_thresholds(mode: str | None = None) -> dict[str, float]:
    return DEFAULT_F2_THRESHOLDS.get(mode or MODE, DEFAULT_F2_THRESHOLDS["multimodal"])
