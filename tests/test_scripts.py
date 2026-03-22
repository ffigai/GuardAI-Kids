import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import pandas as pd

from scripts.backfill_thumbnail_labels import apply_thumbnail_labels


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.artifact_dir = Path("artifacts") / "test_policy_script"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("artifacts") / "test_thumbnail_label_script"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "text",
            "label_order": ["ADD", "SXL", "PH", "HH"],
            "video_ids": ["vid-1", "vid-2"],
            "texts": ["safe text", "unsafe text"],
            "labels": [[0, 0, 0, 0], [1, 0, 0, 0]],
            "logits": [[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
            "predictions": [[0.05, 0.02, 0.01, 0.01], [0.99, 0.01, 0.01, 0.01]],
        }
        (self.artifact_dir / "predictions_text.json").write_text(json.dumps(payload), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.artifact_dir, ignore_errors=True)
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def test_reevaluate_policy_script_writes_output(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path("src").resolve())
        subprocess.run(
            [
                str(Path(".venv") / "Scripts" / "python.exe"),
                "scripts/reevaluate_policy_from_predictions.py",
                "--artifact-dir",
                str(self.artifact_dir),
                "--mode",
                "text",
            ],
            check=True,
            env=env,
        )
        output_path = self.artifact_dir / "policy_eval_text.json"
        self.assertTrue(output_path.exists())

    def test_apply_thumbnail_labels_copies_harm_categories(self):
        source_df = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "harm_cat": ["ADD", ""],
            }
        )
        enriched_df = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "thumbnail_status": ["downloaded", "missing"],
            }
        )

        updated = apply_thumbnail_labels(source_df, enriched_df)

        self.assertEqual(updated["thumbnail_harm_cat"].tolist(), ["ADD", ""])

    def test_backfill_thumbnail_labels_script_updates_existing_workbooks(self):
        harmful_source = pd.DataFrame(
            {
                "video_id": ["v1"],
                "harm_cat": ["SXL"],
                "title": ["demo"],
                "description": [""],
                "transcript": [""],
            }
        )
        harmless_source = pd.DataFrame(
            {
                "video_id": ["v2"],
                "harm_cat": [""],
                "title": ["safe"],
                "description": [""],
                "transcript": [""],
            }
        )
        harmful_enriched = pd.DataFrame(
            {
                "video_id": ["v1"],
                "thumbnail_status": ["downloaded"],
            }
        )
        harmless_enriched = pd.DataFrame(
            {
                "video_id": ["v2"],
                "thumbnail_status": ["missing"],
            }
        )

        harmful_source.to_excel(self.data_dir / "Harmful.xlsx", index=False)
        harmless_source.to_excel(self.data_dir / "Harmless.xlsx", index=False)
        harmful_enriched.to_excel(self.data_dir / "Harmful_with_thumbnails.xlsx", index=False)
        harmless_enriched.to_excel(self.data_dir / "Harmless_with_thumbnails.xlsx", index=False)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path("src").resolve())
        subprocess.run(
            [
                str(Path(".venv") / "Scripts" / "python.exe"),
                "scripts/backfill_thumbnail_labels.py",
                "--data-dir",
                str(self.data_dir),
            ],
            check=True,
            env=env,
        )

        updated_harmful = pd.read_excel(self.data_dir / "Harmful_with_thumbnails.xlsx")
        updated_harmless = pd.read_excel(self.data_dir / "Harmless_with_thumbnails.xlsx")
        self.assertEqual(updated_harmful.loc[0, "thumbnail_harm_cat"], "SXL")
        self.assertTrue(pd.isna(updated_harmless.loc[0, "thumbnail_harm_cat"]) or updated_harmless.loc[0, "thumbnail_harm_cat"] == "")


if __name__ == "__main__":
    unittest.main()
