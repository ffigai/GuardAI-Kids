"""Runtime thumbnail feature extraction for image and multimodal inference."""

from __future__ import annotations

import io
from functools import lru_cache
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPModel, CLIPProcessor

from guardaikids.config import (
    IMAGE_FEATURE_DIMS,
    IMAGE_ANALYSIS_MODEL,
    IMAGE_QUALITY_FEATURE_NAMES,
    IMAGE_SIMILARITY_PROMPTS,
)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
NSFW_MODEL_NAME = "Marqo/nsfw-image-detection-384"
VIOLENCE_MODEL_NAME = "jaranohaal/vit-base-violence-detection"
SUPPORTED_IMAGE_ANALYSIS_MODELS = {"clip", "clip_ocr", "clip_nsfw_violence"}
OCR_CONFIDENCE_THRESHOLD = 0.4
OCR_MIN_TEXT_LENGTH = 15  # ignore OCR hits shorter than this (UI chrome: "HD", "4K", channel handles, etc.)


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
    expected_dim = IMAGE_FEATURE_DIMS["clip"]
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
    if feature_vector.shape != (expected_dim,):
        raise ValueError(f"Expected feature vector of shape ({expected_dim},), got {feature_vector.shape}")
    return feature_vector


@lru_cache(maxsize=1)
def _load_ocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())


def _extract_ocr_text(image: Image.Image) -> str:
    reader = _load_ocr_reader()
    results = reader.readtext(np.asarray(image))
    words = [text for _, text, confidence in results if confidence >= OCR_CONFIDENCE_THRESHOLD]
    combined = " ".join(words)
    return combined if len(combined) >= OCR_MIN_TEXT_LENGTH else ""


def build_feature_vector_from_image_with_ocr(image: Image.Image, model_name: str = CLIP_MODEL_NAME) -> np.ndarray:
    expected_dim = IMAGE_FEATURE_DIMS["clip_ocr"]
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

    ocr_text = _extract_ocr_text(image)
    has_ocr_text = 1.0 if ocr_text.strip() else 0.0
    if ocr_text.strip():
        ocr_inputs = processor(text=[ocr_text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(_clip_device())
        with torch.no_grad():
            ocr_text_output = model.get_text_features(**ocr_inputs)
        ocr_text_features = normalize_tensor(_extract_feature_tensor(ocr_text_output, "text_embeds"))
        with torch.no_grad():
            ocr_similarity_scores = torch.matmul(ocr_text_features, text_features.T)[0]
        ocr_scores = ocr_similarity_scores.detach().cpu().numpy().astype(np.float32)
    else:
        ocr_scores = np.zeros(len(IMAGE_SIMILARITY_PROMPTS), dtype=np.float32)

    feature_vector = np.concatenate(
        [
            image_features[0].detach().cpu().numpy().astype(np.float32),
            similarity_scores.detach().cpu().numpy().astype(np.float32),
            quality_features,
            np.array([0.0], dtype=np.float32),
            ocr_scores,
            np.array([has_ocr_text], dtype=np.float32),
        ]
    )
    if feature_vector.shape != (expected_dim,):
        raise ValueError(f"Expected feature vector of shape ({expected_dim},), got {feature_vector.shape}")
    return feature_vector


@lru_cache(maxsize=1)
def _load_nsfw_classifier(model_name: str = NSFW_MODEL_NAME):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(_clip_device())
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _load_violence_classifier(model_name: str = VIOLENCE_MODEL_NAME):
    """Load the timm-format violence classifier from HuggingFace Hub.

    The checkpoint uses timm ViT-B/16 weight naming but lacks hub metadata for
    either timm or HuggingFace Auto classes. We build the architecture manually
    and download the weights directly.
    """
    import timm
    from timm.data import create_transform, resolve_model_data_config
    from huggingface_hub import hf_hub_download

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)

    try:
        weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    except Exception:
        weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict)
    model = model.to(_clip_device())
    model.eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return transform, model


def _score_violence_classifier(image: Image.Image, transform, model, violent_class_idx: int = 1) -> float:
    """Return P(violent) from the timm violence classifier.

    Class ordering follows RLVS dataset convention: 0=NonViolence, 1=Violence.
    """
    tensor = transform(image).unsqueeze(0).to(_clip_device())
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[violent_class_idx].cpu())


def _score_binary_classifier(image: Image.Image, processor, model, positive_label: str) -> float:
    """Return the softmax probability for `positive_label` from a binary image classifier."""
    inputs = processor(images=image, return_tensors="pt").to(_clip_device())
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label_map = {v.lower(): i for i, v in model.config.id2label.items()}
    idx = label_map.get(positive_label.lower())
    if idx is None:
        raise ValueError(f"Label '{positive_label}' not found in model. Available: {list(model.config.id2label.values())}")
    return float(probs[idx].cpu())


def build_feature_vector_from_image_with_specialists(image: Image.Image, clip_model_name: str = CLIP_MODEL_NAME) -> np.ndarray:
    """Build feature vector using CLIP + NSFW classifier + violence classifier + gated OCR harm scores.

    Layout:
      [0:512]    CLIP image embedding
      [512:520]  CLIP similarity scores (8 harm prompts)
      [520:525]  image quality features (5)
      [525]      NSFW score (Marqo/nsfw-image-detection-384 classifier)
      [526]      violence score (jaranohaal/vit-base-violence-detection classifier)
      [527]      has_ocr_text flag (1.0 if text detected, else 0.0)
      [528:536]  OCR harm similarity scores (8) — zeroed when has_ocr_text=0
      [536]      missing thumbnail flag (always 0.0 here)
    """
    expected_dim = IMAGE_FEATURE_DIMS["clip_nsfw_violence"]

    # --- CLIP visual embedding + harm similarity scores ---
    processor, clip_model = _load_clip_components(clip_model_name)
    text_features = _get_text_features(clip_model_name)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_clip_device())
    with torch.no_grad():
        image_output = clip_model.get_image_features(pixel_values=pixel_values)
    image_features = normalize_tensor(_extract_feature_tensor(image_output, "image_embeds"))
    with torch.no_grad():
        similarity_scores = torch.matmul(image_features, text_features.T)[0]

    # --- image quality ---
    quality_features = _compute_quality_features(image)

    # --- specialist scores ---
    nsfw_processor, nsfw_model = _load_nsfw_classifier()
    nsfw_score = _score_binary_classifier(image, nsfw_processor, nsfw_model, positive_label="nsfw")

    violence_transform, violence_model = _load_violence_classifier()
    violence_score = _score_violence_classifier(image, violence_transform, violence_model)

    # --- gated OCR harm scores ---
    ocr_text = _extract_ocr_text(image)
    has_ocr_text = 1.0 if ocr_text.strip() else 0.0
    if ocr_text.strip():
        ocr_inputs = processor(
            text=[ocr_text], return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(_clip_device())
        with torch.no_grad():
            ocr_text_output = clip_model.get_text_features(**ocr_inputs)
        ocr_text_features = normalize_tensor(_extract_feature_tensor(ocr_text_output, "text_embeds"))
        with torch.no_grad():
            ocr_similarity_scores = torch.matmul(ocr_text_features, text_features.T)[0]
        ocr_scores = ocr_similarity_scores.detach().cpu().numpy().astype(np.float32)
    else:
        ocr_scores = np.zeros(len(IMAGE_SIMILARITY_PROMPTS), dtype=np.float32)

    feature_vector = np.concatenate([
        image_features[0].detach().cpu().numpy().astype(np.float32),
        similarity_scores.detach().cpu().numpy().astype(np.float32),
        quality_features,
        np.array([nsfw_score, violence_score, has_ocr_text], dtype=np.float32),
        ocr_scores,
        np.array([0.0], dtype=np.float32),  # missing thumbnail flag
    ])
    if feature_vector.shape != (expected_dim,):
        raise ValueError(f"Expected feature vector of shape ({expected_dim},), got {feature_vector.shape}")
    return feature_vector


def extract_thumbnail_features_from_url(
    url: str,
    model_name: str = CLIP_MODEL_NAME,
    image_analysis_model: str | None = None,
) -> np.ndarray:
    selected_model = get_image_analysis_model(image_analysis_model)
    image = _load_image_from_url(url)
    if selected_model == "clip_nsfw_violence":
        return build_feature_vector_from_image_with_specialists(image, clip_model_name=model_name)
    if selected_model == "clip_ocr":
        return build_feature_vector_from_image_with_ocr(image, model_name=model_name)
    return build_feature_vector_from_image(image, model_name=model_name)
