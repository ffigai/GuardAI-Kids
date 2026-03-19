import unittest

import numpy as np
import pandas as pd

from etp.policy import evaluate_policy


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


if __name__ == "__main__":
    unittest.main()
