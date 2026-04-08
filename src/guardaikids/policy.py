"""Policy rules and evaluation helpers."""

import numpy as np
import pandas as pd

from guardaikids.config import LABELS_ORDER, MODE, get_default_thresholds


def rule_based_decision(probs_row, thresholds: dict[str, float]) -> tuple[str, list[str]]:
    """Return (decision, fired_categories) given a probability row and flat thresholds."""
    fired = [
        label
        for index, label in enumerate(LABELS_ORDER)
        if probs_row[index] >= thresholds[label]
    ]
    return ("Harmful", fired) if fired else ("Safe", [])


def build_decision_dataframe(probs: np.ndarray, thresholds: dict[str, float] | None = None, mode: str = MODE) -> pd.DataFrame:
    resolved = thresholds or get_default_thresholds(mode)
    decisions, categories = zip(*[rule_based_decision(row, resolved) for row in probs])
    return pd.DataFrame({"decision": decisions, "categories": list(categories)})


def get_policy_decision(probs_row, thresholds: dict[str, float] | None = None, mode: str = MODE) -> dict[str, object]:
    resolved = thresholds or get_default_thresholds(mode)
    fired = []
    for index, label in enumerate(LABELS_ORDER):
        if probs_row[index] >= resolved[label]:
            fired.append({"label": label, "probability": float(probs_row[index]), "threshold": float(resolved[label])})

    if fired:
        fired.sort(key=lambda x: x["probability"], reverse=True)
        return {
            "decision": "Harmful",
            "categories": [item["label"] for item in fired],
            "trigger_category": fired[0]["label"],
            "probability": fired[0]["probability"],
            "threshold": fired[0]["threshold"],
        }
    return {
        "decision": "Safe",
        "categories": [],
        "trigger_category": None,
        "probability": None,
        "threshold": None,
    }


def evaluate_policy(decision_df: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    harmful = (labels.sum(axis=1) > 0).astype(int)
    harmful_total = int(harmful.sum())
    decisions = decision_df["decision"].values
    flagged = (decisions == "Harmful").astype(int)
    allowed = (decisions == "Safe").astype(int)
    tp = int(np.sum((flagged == 1) & (harmful == 1)))
    fp = int(np.sum((flagged == 1) & (harmful == 0)))
    fn = int(np.sum((allowed == 1) & (harmful == 1)))
    tn = int(np.sum((allowed == 1) & (harmful == 0)))
    return {
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / harmful_total if harmful_total else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / harmful_total if harmful_total else 0.0,
    }


def evaluate_protection(decision_df: pd.DataFrame, labels: np.ndarray) -> float:
    harmful = (labels.sum(axis=1) > 0).astype(int)
    flagged = (decision_df["decision"].values == "Harmful").astype(int)
    tp = int(np.sum((flagged == 1) & (harmful == 1)))
    fn = int(np.sum((flagged == 0) & (harmful == 1)))
    return tp / (tp + fn) if (tp + fn) else 0.0
