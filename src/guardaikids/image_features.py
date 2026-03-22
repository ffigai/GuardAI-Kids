"""Runtime thumbnail feature extraction for image and multimodal inference."""

from __future__ import annotations

import io
from functools import lru_cache
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from guardaikids.config import (
    IMAGE_FEATURE_DIM,
    IMAGE_ANALYSIS_MODEL,
    IMAGE_QUALITY_FEATURE_NAMES,
    IMAGE_SIMILARITY_PROMPTS,
)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SUPPORTED_IMAGE_ANALYSIS_MODELS = {"clip"}


def get_image_analysis_model(model_name: str | None = None) -> str:
    selected = model_name or IMAGE_ANALYSIS_MODEL
    if selected not in SUPPORTED_IMAGE_ANALYSIS_MODELS:
        raise ValueError(f"Unsupported image analysis model: {selected}")
    return selected


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True)


def _extract_feature_tensor(output, embed_attr: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, embed_attr):
        embeds = getattr(output, embed_attr)
        if embeds is not None:
            return embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")


def _clip_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_clip_components(model_name: str = CLIP_MODEL_NAME):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(_clip_device())
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _get_text_features(model_name: str = CLIP_MODEL_NAME) -> torch.Tensor:
    processor, model = _load_clip_components(model_name)
    text_inputs = processor(text=IMAGE_SIMILARITY_PROMPTS, return_tensors="pt", padding=True).to(_clip_device())
    with torch.no_grad():
        output = model.get_text_features(**text_inputs)
    return normalize_tensor(_extract_feature_tensor(output, "text_embeds"))


def _compute_quality_features(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    brightness_mean = float(grayscale.mean())
    contrast_std = float(grayscale.std())
    saturation_mean = float(hsv[..., 1].mean())

    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    horizontal_detail = np.abs(np.diff(grayscale, axis=1)).mean() if grayscale.shape[1] > 1 else 0.0
    vertical_detail = np.abs(np.diff(grayscale, axis=0)).mean() if grayscale.shape[0] > 1 else 0.0
    detail_score = float(horizontal_detail + vertical_detail)

    features = np.array(
        [
            brightness_mean,
            contrast_std,
            saturation_mean,
            colorfulness,
            detail_score,
        ],
        dtype=np.float32,
    )
    if features.shape != (len(IMAGE_QUALITY_FEATURE_NAMES),):
        raise ValueError(
            f"Expected {len(IMAGE_QUALITY_FEATURE_NAMES)} image quality features, got {features.shape}"
        )
    return features


def _load_image_from_url(url: str) -> Image.Image:
    try:
        with urlopen(url, timeout=20) as response:
            image_bytes = response.read()
    except URLError as exc:
        raise ValueError(f"Failed to download thumbnail from {url}") from exc
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def build_feature_vector_from_image(image: Image.Image, model_name: str = CLIP_MODEL_NAME) -> np.ndarray:
    get_image_analysis_model("clip")
    processor, model = _load_clip_components(model_name)
    text_features = _get_text_features(model_name)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_clip_device())

    with torch.no_grad():
        image_output = model.get_image_features(pixel_values=pixel_values)
    image_features = normalize_tensor(_extract_feature_tensor(image_output, "image_embeds"))
    with torch.no_grad():
        similarity_scores = torch.matmul(image_features, text_features.T)[0]

    quality_features = _compute_quality_features(image)
    feature_vector = np.concatenate(
        [
            image_features[0].detach().cpu().numpy().astype(np.float32),
            similarity_scores.detach().cpu().numpy().astype(np.float32),
            quality_features,
            np.array([0.0], dtype=np.float32),
        ]
    )
    if feature_vector.shape != (IMAGE_FEATURE_DIM,):
        raise ValueError(f"Expected feature vector of shape ({IMAGE_FEATURE_DIM},), got {feature_vector.shape}")
    return feature_vector


def extract_thumbnail_features_from_url(
    url: str,
    model_name: str = CLIP_MODEL_NAME,
    image_analysis_model: str | None = None,
) -> np.ndarray:
    selected_model = get_image_analysis_model(image_analysis_model)
    if selected_model != "clip":
        raise ValueError(f"Unsupported image analysis model: {selected_model}")
    image = _load_image_from_url(url)
    return build_feature_vector_from_image(image, model_name=model_name)
