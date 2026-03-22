import unittest

import numpy as np
import pandas as pd

from guardaikids.policy import build_decision_dataframe, evaluate_policy


class PolicyEvaluationTests(unittest.TestCase):
    def test_warn_is_not_counted_as_false_allow(self):
        decision_df = pd.DataFrame({"0_4": ["Warn", "Allow", "Block"]})
        labels = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )

        metrics = evaluate_policy(decision_df, labels)["0_4"]

        self.assertAlmostEqual(metrics["false_allow_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["block_recall"], 1 / 3)
        self.assertAlmostEqual(metrics["protection_precision"], 1.0)

    def test_image_mode_uses_more_conservative_warn_thresholds(self):
        probs = np.array([[0.11, 0.11, 0.11, 0.11]])

        text_decisions = build_decision_dataframe(probs, mode="text")
        image_decisions = build_decision_dataframe(probs, mode="image")

        self.assertEqual(text_decisions.loc[0, "0_4"], "Warn")
        self.assertEqual(image_decisions.loc[0, "0_4"], "Allow")


if __name__ == "__main__":
    unittest.main()
