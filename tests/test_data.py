import unittest

import pandas as pd

from etp.data import split_train_validation, validate_source_columns


class DataValidationTests(unittest.TestCase):
    def test_validate_source_columns_reports_missing_columns(self):
        df = pd.DataFrame({"title": ["a"], "description": ["b"]})

        with self.assertRaisesRegex(ValueError, "harm_cat"):
            validate_source_columns(df)

    def test_split_train_validation_rejects_missing_positive_labels_in_validation(self):
        model_df = pd.DataFrame(
            {
                "text": [f"sample {index}" for index in range(5)],
                "ADD": [1, 0, 0, 0, 0],
                "SXL": [1, 1, 0, 0, 0],
                "PH": [1, 1, 0, 0, 0],
                "HH": [1, 1, 0, 0, 0],
            }
        )

        with self.assertRaisesRegex(ValueError, "Validation split is missing positive samples"):
            split_train_validation(model_df, test_size=0.2, random_state=42)


if __name__ == "__main__":
    unittest.main()
