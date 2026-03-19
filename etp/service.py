"""Product-facing training and analysis services."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from etp.config import AGE_GROUPS, LABELS_ORDER, default_artifact_dir
from etp.explainability import explain_video
from etp.modeling import load_saved_model, predict_video_text
from etp.workflow import run_training_workflow, save_training_artifacts
from etp.youtube import build_model_input, build_youtube_client, fetch_youtube_metadata


@dataclass
class AnalysisArtifacts:
    model: object
    tokenizer: object
    metadata: dict[str, object]
    artifact_dir: Path


def resolve_artifact_dir(path: str | Path | None = None) -> Path:
    configured = path or os.environ.get("ETP_ARTIFACT_DIR")
    return Path(configured) if configured else default_artifact_dir()


def train_and_save_system(
    harmful_path: str | Path,
    harmless_path: str | Path,
    artifact_dir: str | Path | None = None,
) -> tuple[dict[str, object], Path]:
    results = run_training_workflow(harmful_path, harmless_path)
    resolved_dir = resolve_artifact_dir(artifact_dir)
    save_training_artifacts(results, resolved_dir)
    return results, resolved_dir


def load_analysis_artifacts(artifact_dir: str | Path | None = None) -> AnalysisArtifacts:
    resolved_dir = resolve_artifact_dir(artifact_dir)
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
) -> dict[str, object]:
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is required to analyze a YouTube URL.")

    artifacts = load_analysis_artifacts(artifact_dir)
    youtube_client = build_youtube_client(api_key)
    metadata = fetch_youtube_metadata(url, youtube_client)
    if metadata is None:
        raise ValueError("Video not found or the URL is invalid.")

    text = build_model_input(metadata)
    probabilities = predict_video_text(artifacts.model, artifacts.tokenizer, text)
    recommendations = {}
    for age_group in AGE_GROUPS:
        explanation = explain_video(text, probabilities, age_group, artifacts.model, artifacts.tokenizer)
        recommendations[age_group] = {
            "decision": explanation["decision"],
            "trigger_category": explanation["trigger_category"],
            "probability": explanation["probability"],
            "threshold": explanation["threshold"],
            "explanation": explanation["human_readable_explanation"],
            "top_tokens": [
                {"token": token, "importance": float(score)} for token, score in explanation["top_tokens"]
            ],
        }

    return {
        "url": url,
        "artifact_dir": str(artifacts.artifact_dir),
        "model_scores": {label: float(probabilities[idx]) for idx, label in enumerate(LABELS_ORDER)},
        "metadata": metadata,
        "recommendations": recommendations,
        "training_metadata": artifacts.metadata,
    }
