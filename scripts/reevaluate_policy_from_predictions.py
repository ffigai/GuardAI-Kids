"""Recompute policy metrics from saved prediction artifacts without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardaikids.config import MODE, default_artifact_dir
from guardaikids.policy import build_decision_dataframe, evaluate_policy, evaluate_protection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute age-aware policy metrics from saved model predictions."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=default_artifact_dir() / MODE,
        help="Artifact directory containing predictions_<mode>.json.",
    )
    parser.add_argument(
        "--mode",
        default=MODE,
        help="Mode name used in the predictions file name.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions_path = args.artifact_dir / f"predictions_{args.mode}.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction artifact not found: {predictions_path}")

    with predictions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    probs = np.array(payload["predictions"], dtype=float)
    labels = np.array(payload["labels"], dtype=int)
    decision_df = build_decision_dataframe(probs, mode=args.mode)
    policy_metrics = evaluate_policy(decision_df, labels)
    protection_metrics = evaluate_protection(decision_df, labels)

    output = {
        "mode": args.mode,
        "artifact_dir": str(args.artifact_dir),
        "policy_metrics": policy_metrics,
        "protection_metrics": protection_metrics,
    }
    output_path = args.artifact_dir / f"policy_eval_{args.mode}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Saved policy evaluation to: {output_path}")


if __name__ == "__main__":
    main()
