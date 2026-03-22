import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from guardaikids.workflow import save_validation_predictions


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("artifacts") / "test_workflow_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_save_validation_predictions_includes_reusable_analysis_fields(self):
        results = {
            "validation_outputs": {
                "labels": np.array([[1, 0, 0, 0]]),
                "logits": np.array([[0.1, 0.2, 0.3, 0.4]]),
                "probs": np.array([[0.5, 0.6, 0.7, 0.8]]),
            },
            "val_df": pd.DataFrame({"video_id": ["abc123"], "text": ["sample text"]}),
        }

        save_validation_predictions(results, self.output_dir, mode="text")

        payload = json.loads((self.output_dir / "predictions_text.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "text")
        self.assertEqual(payload["video_ids"], ["abc123"])
        self.assertEqual(payload["texts"], ["sample text"])
        self.assertIn("logits", payload)
        self.assertIn("predictions", payload)
        self.assertIn("labels", payload)


if __name__ == "__main__":
    unittest.main()
