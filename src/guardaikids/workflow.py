"""Orchestration helpers for training and evaluation workflows."""

import json
from pathlib import Path

import numpy as np

from guardaikids.config import IMAGE_ANALYSIS_MODEL, IMAGE_FEATURE_DIMS, LABELS_ORDER, MODE, MODEL_NAME
from guardaikids.data import load_raw_data, prepare_model_dataframe, split_train_validation, to_hf_dataset
from guardaikids.modeling import (
    apply_thresholds,
    collect_validation_outputs,
    optimize_thresholds,
    summarize_validation_metrics,
    train_multilabel_classifier,
)
from guardaikids.policy import build_decision_dataframe, evaluate_policy, evaluate_protection


def _to_jsonable(value):
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_validation_predictions(results: dict[str, object], artifact_dir: str | Path, mode: str = MODE) -> None:
    resolved_dir = Path(artifact_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    outputs = results["validation_outputs"]
    val_df = results["val_df"]
    payload = {
        "mode": mode,
        "label_order": LABELS_ORDER,
        "video_ids": val_df["video_id"].tolist() if "video_id" in val_df.columns else [],
        "texts": val_df["text"].tolist() if "text" in val_df.columns else [],
        "labels": outputs["labels"].tolist(),
        "logits": outputs["logits"].tolist(),
        "predictions": outputs["probs"].tolist(),
    }
    output_path = resolved_dir / f"predictions_{mode}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2)


def run_training_workflow(
    harmful_path: str | Path,
    harmless_path: str | Path,
    mode: str = MODE,
    image_analysis_model: str | None = None,
    image_feature_dir: str | Path | None = None,
) -> dict[str, object]:
    selected_image_model = image_analysis_model or IMAGE_ANALYSIS_MODEL
    image_feature_dim = IMAGE_FEATURE_DIMS[selected_image_model]
    raw_df = load_raw_data(harmful_path, harmless_path)
    model_df = prepare_model_dataframe(raw_df, LABELS_ORDER, mode=mode)
    train_df, val_df = split_train_validation(model_df, LABELS_ORDER)

    artifacts = train_multilabel_classifier(
        to_hf_dataset(train_df),
        to_hf_dataset(val_df),
        mode=mode,
        image_feature_dim=image_feature_dim,
        image_feature_dir=image_feature_dir,
    )
    outputs = collect_validation_outputs(artifacts.trainer, artifacts.val_dataset)

    default_summary = summarize_validation_metrics(outputs["labels"], outputs["probs"], outputs["preds"])
    f1_thresholds = optimize_thresholds(outputs["labels"], outputs["probs"], beta=1.0)
    f2_thresholds = optimize_thresholds(outputs["labels"], outputs["probs"], beta=2.0)
    tuned_predictions = apply_thresholds(outputs["probs"], f2_thresholds)
    tuned_summary = summarize_validation_metrics(outputs["labels"], outputs["probs"], tuned_predictions)

    decision_df = build_decision_dataframe(outputs["probs"], thresholds=f2_thresholds, mode=mode)
    policy_metrics = evaluate_policy(decision_df, outputs["labels"])
    protection_metrics = evaluate_protection(decision_df, outputs["labels"])

    return {
        "raw_df": raw_df,
        "model_df": model_df,
        "train_df": train_df,
        "val_df": val_df,
        "artifacts": artifacts,
        "validation_outputs": outputs,
        "default_summary": default_summary,
        "f1_thresholds": f1_thresholds,
        "f2_thresholds": f2_thresholds,
        "tuned_summary": tuned_summary,
        "decision_df": decision_df,
        "policy_metrics": policy_metrics,
        "protection_metrics": protection_metrics,
    }


def save_training_artifacts(
    results: dict[str, object],
    artifact_dir: str | Path,
    mode: str = MODE,
    image_analysis_model: str | None = None,
    xai_method: str | None = None,
) -> None:
    resolved_dir = Path(artifact_dir)
    model_dir = resolved_dir / "model"
    tokenizer_dir = resolved_dir / "tokenizer"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    artifacts = results["artifacts"]
    artifacts.model.save_pretrained(model_dir)
    artifacts.tokenizer.save_pretrained(tokenizer_dir)

    selected_image_model = image_analysis_model or IMAGE_ANALYSIS_MODEL
    metadata = {
        "model_name": MODEL_NAME,
        "mode": mode,
        "image_feature_dim": IMAGE_FEATURE_DIMS[selected_image_model],
        "image_analysis_model": selected_image_model,
        "xai_method": xai_method,
        "train_size": int(len(results["train_df"])),
        "validation_size": int(len(results["val_df"])),
        "f1_thresholds": results["f1_thresholds"],
        "f2_thresholds": results["f2_thresholds"],
        "policy_metrics": results["policy_metrics"],
        "protection_metrics": results["protection_metrics"],
        "roc_auc": results["default_summary"]["roc_auc"],
        "positives": results["default_summary"]["positives"],
    }
    with (resolved_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(metadata), handle, indent=2)
    save_validation_predictions(results, resolved_dir, mode=mode)
