import unittest
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from datasets import Dataset

from guardaikids.config import IMAGE_FEATURE_DIM
from guardaikids.data import (
    load_image_features,
    prepare_model_dataframe,
    prepare_dataset_inputs,
    split_train_validation,
    validate_source_columns,
)


class DataValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path("artifacts") / "test_tmp_data"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_validate_source_columns_reports_missing_columns(self):
        df = pd.DataFrame({"title": ["a"], "description": ["b"]})

        with self.assertRaisesRegex(ValueError, "harm_cat"):
            validate_source_columns(df)

    def test_split_train_validation_rejects_missing_positive_labels_in_validation(self):
        model_df = pd.DataFrame(
            {
                "video_id": [f"id-{index}" for index in range(5)],
                "text": [f"sample {index}" for index in range(5)],
                "ADD": [1, 0, 0, 0, 0],
                "SXL": [1, 1, 0, 0, 0],
                "PH": [1, 1, 0, 0, 0],
                "HH": [1, 1, 0, 0, 0],
            }
        )

        with self.assertRaisesRegex(ValueError, "Validation split is missing positive samples"):
            split_train_validation(model_df, test_size=0.2, random_state=42)

    def test_load_image_features_marks_missing_files(self):
        features = load_image_features("missing-video", image_feature_dir=self.temp_root)
        self.assertEqual(len(features), IMAGE_FEATURE_DIM)
        self.assertEqual(features[-1], 1.0)
        self.assertTrue(all(value == 0.0 for value in features[:-1]))

    def test_prepare_dataset_inputs_adds_image_features_for_multimodal_mode(self):
        existing = np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)
        np.save(self.temp_root / "video-1.npy", existing)
        dataset = Dataset.from_dict(
            {
                "video_id": ["video-1", "video-2"],
                "text": ["hello", "world"],
                "labels": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            }
        )
        prepared = prepare_dataset_inputs(dataset, mode="multimodal", image_feature_dir=self.temp_root)

        self.assertIn("image_features", prepared.column_names)
        self.assertEqual(len(prepared[0]["image_features"]), IMAGE_FEATURE_DIM)
        self.assertEqual(prepared[0]["image_features"][-1], 0.0)
        self.assertEqual(prepared[1]["image_features"][-1], 1.0)

    def test_load_image_features_falls_back_when_shape_is_invalid(self):
        np.save(self.temp_root / "video-bad.npy", np.zeros(3, dtype=np.float32))

        features = load_image_features("video-bad", image_feature_dir=self.temp_root)

        self.assertEqual(len(features), IMAGE_FEATURE_DIM)
        self.assertEqual(features[-1], 1.0)

    def test_prepare_model_dataframe_uses_thumbnail_labels_for_image_mode_when_available(self):
        df = pd.DataFrame(
            {
                "video_id": ["video-1", "video-2"],
                "harm_cat": ["ADD", ""],
                "thumbnail_harm_cat": ["SXL", ""],
                "title": ["title 1", "title 2"],
                "description": ["desc 1", "desc 2"],
                "transcript": ["transcript 1", "transcript 2"],
            }
        )

        prepared = prepare_model_dataframe(df, mode="image")

        self.assertEqual(prepared.loc[0, "label_source"], "thumbnail")
        self.assertEqual(prepared.loc[0, "ADD"], 0)
        self.assertEqual(prepared.loc[0, "SXL"], 1)
        self.assertEqual(prepared.loc[1, "label_source"], "video")


if __name__ == "__main__":
    unittest.main()
