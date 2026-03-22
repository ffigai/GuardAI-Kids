"""Age-aware policy rules and evaluation helpers."""

import numpy as np
import pandas as pd

from guardaikids.config import LABELS_ORDER, MODE, get_default_thresholds


def rule_based_decision(probs_row, age_group: str, thresholds=None, mode: str | None = None) -> str:
    threshold_map = thresholds or get_default_thresholds(mode)
    age_thresholds = threshold_map[age_group]

    for index, label in enumerate(LABELS_ORDER):
        if probs_row[index] >= age_thresholds[label]["block"]:
            return "Block"

    for index, label in enumerate(LABELS_ORDER):
        if probs_row[index] >= age_thresholds[label]["warn"]:
            return "Warn"

    return "Allow"


def build_decision_dataframe(probs: np.ndarray, thresholds=None, mode: str = MODE) -> pd.DataFrame:
    threshold_map = thresholds or get_default_thresholds(mode)
    decision_df = pd.DataFrame()
    for age_group in threshold_map:
        decision_df[age_group] = [rule_based_decision(row, age_group, threshold_map, mode=mode) for row in probs]
    return decision_df


def get_policy_decision(probs_row, age_group: str, thresholds=None, mode: str = MODE) -> dict[str, object]:
    threshold_map = thresholds or get_default_thresholds(mode)
    age_thresholds = threshold_map[age_group]

    for index, label in enumerate(LABELS_ORDER):
        probability = probs_row[index]
        block_threshold = age_thresholds[label]["block"]
        if probability >= block_threshold:
            return {
                "decision": "Block",
                "category": label,
                "probability": float(probability),
                "threshold": block_threshold,
            }

    for index, label in enumerate(LABELS_ORDER):
        probability = probs_row[index]
        warn_threshold = age_thresholds[label]["warn"]
        if probability >= warn_threshold:
            return {
                "decision": "Warn",
                "category": label,
                "probability": float(probability),
                "threshold": warn_threshold,
            }

    return {
        "decision": "Allow",
        "category": None,
        "probability": None,
        "threshold": None,
    }


def evaluate_policy(decision_df: pd.DataFrame, labels: np.ndarray) -> dict[str, dict[str, float]]:
    harmful = (labels.sum(axis=1) > 0).astype(int)
    harmful_total = int(harmful.sum())
    results = {}
    for age_group in decision_df.columns:
        decisions = decision_df[age_group].values
        blocked = (decisions == "Block").astype(int)
        protected = ((decisions == "Block") | (decisions == "Warn")).astype(int)
        allowed = (decisions == "Allow").astype(int)
        tp = np.sum((blocked == 1) & (harmful == 1))
        fp = np.sum((blocked == 1) & (harmful == 0))
        fn = np.sum((allowed == 1) & (harmful == 1))
        tn = np.sum((allowed == 1) & (harmful == 0))
        protected_tp = np.sum((protected == 1) & (harmful == 1))
        protected_fp = np.sum((protected == 1) & (harmful == 0))
        results[age_group] = {
            "block_precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "block_recall": tp / harmful_total if harmful_total else 0.0,
            "false_block_rate": fp / (fp + tn) if (fp + tn) else 0.0,
            "false_allow_rate": fn / harmful_total if harmful_total else 0.0,
            "protection_precision": protected_tp / (protected_tp + protected_fp)
            if (protected_tp + protected_fp)
            else 0.0,
        }
    return results


def evaluate_protection(decision_df: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    harmful = (labels.sum(axis=1) > 0).astype(int)
    results = {}
    for age_group in decision_df.columns:
        decisions = decision_df[age_group].values
        protected = ((decisions == "Block") | (decisions == "Warn")).astype(int)
        tp = np.sum((protected == 1) & (harmful == 1))
        fn = np.sum((protected == 0) & (harmful == 1))
        results[age_group] = tp / (tp + fn) if (tp + fn) else 0.0
    return results
