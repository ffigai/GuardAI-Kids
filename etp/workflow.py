"""Orchestration helpers for training and evaluation workflows."""

import json
from pathlib import Path

import numpy as np

from etp.config import LABELS_ORDER, MODEL_NAME
from etp.data import load_raw_data, prepare_model_dataframe, split_train_validation, to_hf_dataset
from etp.modeling import (
    apply_thresholds,
    collect_validation_outputs,
    optimize_thresholds,
    summarize_validation_metrics,
    train_multilabel_classifier,
)
from etp.policy import build_decision_dataframe, evaluate_policy, evaluate_protection


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


def run_training_workflow(harmful_path: str | Path, harmless_path: str | Path) -> dict[str, object]:
    raw_df = load_raw_data(harmful_path, harmless_path)
    model_df = prepare_model_dataframe(raw_df, LABELS_ORDER)
    train_df, val_df = split_train_validation(model_df, LABELS_ORDER)

    artifacts = train_multilabel_classifier(to_hf_dataset(train_df), to_hf_dataset(val_df))
    outputs = collect_validation_outputs(artifacts.trainer, artifacts.val_dataset)

    default_summary = summarize_validation_metrics(outputs["labels"], outputs["probs"], outputs["preds"])
    f1_thresholds = optimize_thresholds(outputs["labels"], outputs["probs"], beta=1.0)
    f2_thresholds = optimize_thresholds(outputs["labels"], outputs["probs"], beta=2.0)
    tuned_predictions = apply_thresholds(outputs["probs"], f2_thresholds)
    tuned_summary = summarize_validation_metrics(outputs["labels"], outputs["probs"], tuned_predictions)

    decision_df = build_decision_dataframe(outputs["probs"])
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


def save_training_artifacts(results: dict[str, object], artifact_dir: str | Path) -> None:
    resolved_dir = Path(artifact_dir)
    model_dir = resolved_dir / "model"
    tokenizer_dir = resolved_dir / "tokenizer"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    artifacts = results["artifacts"]
    artifacts.model.save_pretrained(model_dir)
    artifacts.tokenizer.save_pretrained(tokenizer_dir)

    metadata = {
        "model_name": MODEL_NAME,
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
