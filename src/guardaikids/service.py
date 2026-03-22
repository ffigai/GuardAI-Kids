"""Product-facing training and analysis services."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from guardaikids.config import AGE_GROUPS, IMAGE_ANALYSIS_MODEL, LABELS_ORDER, MODE, XAI_METHOD, default_artifact_dir
from guardaikids.explainability import explain_video
from guardaikids.image_features import extract_thumbnail_features_from_url
from guardaikids.modeling import load_saved_model, predict_video_text
from guardaikids.workflow import run_training_workflow, save_training_artifacts
from guardaikids.youtube import build_model_input, build_youtube_client, fetch_youtube_metadata


@dataclass
class AnalysisArtifacts:
    model: object
    tokenizer: object
    metadata: dict[str, object]
    artifact_dir: Path


def resolve_artifact_dir(path: str | Path | None = None, mode: str | None = None) -> Path:
    configured = path or os.environ.get("ETP_ARTIFACT_DIR")
    return Path(configured) if configured else default_artifact_dir() / (mode or MODE)


def train_and_save_system(
    harmful_path: str | Path,
    harmless_path: str | Path,
    artifact_dir: str | Path | None = None,
    mode: str | None = None,
    image_analysis_model: str | None = None,
    xai_method: str | None = None,
) -> tuple[dict[str, object], Path]:
    selected_mode = mode or MODE
    results = run_training_workflow(harmful_path, harmless_path, mode=selected_mode)
    resolved_dir = resolve_artifact_dir(artifact_dir, mode=selected_mode)
    save_training_artifacts(
        results,
        resolved_dir,
        mode=selected_mode,
        image_analysis_model=image_analysis_model or IMAGE_ANALYSIS_MODEL,
        xai_method=xai_method or XAI_METHOD,
    )
    return results, resolved_dir


def load_analysis_artifacts(artifact_dir: str | Path | None = None, mode: str | None = None) -> AnalysisArtifacts:
    resolved_dir = resolve_artifact_dir(artifact_dir, mode=mode)
    metadata_path = resolved_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Trained artifacts were not found in {resolved_dir}. Run the training workflow first."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    model_dir = resolved_dir / "model"
    tokenizer_dir = resolved_dir / "tokenizer"
    model, tokenizer = load_saved_model(model_dir, tokenizer_dir)
    return AnalysisArtifacts(model=model, tokenizer=tokenizer, metadata=metadata, artifact_dir=resolved_dir)


def analyze_youtube_url(
    url: str,
    api_key: str,
    artifact_dir: str | Path | None = None,
    mode: str | None = None,
    image_analysis_model: str | None = None,
    xai_method: str | None = None,
) -> dict[str, object]:
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is required to analyze a YouTube URL.")

    selected_mode = mode or MODE
    if selected_mode not in {"text", "image", "multimodal"}:
        raise ValueError(f"Unsupported mode: {selected_mode}")

    artifacts = load_analysis_artifacts(artifact_dir, mode=selected_mode)
    youtube_client = build_youtube_client(api_key)
    metadata = fetch_youtube_metadata(url, youtube_client)
    if metadata is None:
        raise ValueError("Video not found or the URL is invalid.")

    text = build_model_input(metadata)
    thumbnail_url = metadata.get("thumbnail_url", "")
    image_features = None
    if selected_mode in {"image", "multimodal"}:
        if not thumbnail_url:
            raise ValueError("Thumbnail was not available for this YouTube video.")
        image_features = extract_thumbnail_features_from_url(
            thumbnail_url,
            image_analysis_model=image_analysis_model or IMAGE_ANALYSIS_MODEL,
        )

    probabilities = predict_video_text(artifacts.model, artifacts.tokenizer, text, image_features=image_features)
    recommendations = {}
    for age_group in AGE_GROUPS:
        explanation = explain_video(
            text,
            probabilities,
            age_group,
            artifacts.model,
            artifacts.tokenizer,
            image_features=image_features,
            xai_method=xai_method or XAI_METHOD,
        )
        recommendations[age_group] = {
            "decision": explanation["decision"],
            "trigger_category": explanation["trigger_category"],
            "probability": explanation["probability"],
            "threshold": explanation["threshold"],
            "explanation": explanation["human_readable_explanation"],
            "explanation_bullets": explanation["explanation_bullets"],
            "top_tokens": [
                {"token": token, "importance": float(score)} for token, score in explanation["top_tokens"]
            ],
            "risk_categories": explanation["risk_categories"],
            "image_highlights": explanation["image_highlights"],
            "modality_summary": explanation["modality_summary"],
        }

    return {
        "url": url,
        "analysis_mode": selected_mode,
        "image_analysis_model": image_analysis_model or IMAGE_ANALYSIS_MODEL,
        "xai_method": xai_method or XAI_METHOD,
        "artifact_dir": str(artifacts.artifact_dir),
        "model_scores": {label: float(probabilities[idx]) for idx, label in enumerate(LABELS_ORDER)},
        "metadata": metadata,
        "recommendations": recommendations,
        "training_metadata": artifacts.metadata,
    }
