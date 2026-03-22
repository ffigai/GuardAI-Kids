import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from guardaikids.config import IMAGE_FEATURE_DIM
from guardaikids.service import analyze_youtube_url, resolve_artifact_dir, train_and_save_system


class ServiceTests(unittest.TestCase):
    def test_resolve_artifact_dir_defaults_to_mode_subdirectory(self):
        with patch("guardaikids.service.MODE", "multimodal"):
            resolved = resolve_artifact_dir()
        self.assertEqual(resolved, Path.cwd() / "artifacts" / "multimodal")

    def test_resolve_artifact_dir_accepts_explicit_mode(self):
        resolved = resolve_artifact_dir(mode="image")
        self.assertEqual(resolved, Path.cwd() / "artifacts" / "image")

    @patch("guardaikids.service.save_training_artifacts")
    @patch("guardaikids.service.run_training_workflow")
    def test_train_and_save_system_uses_explicit_mode(self, run_training_workflow, save_training_artifacts):
        run_training_workflow.return_value = {"train_df": [], "val_df": []}

        _results, artifact_dir = train_and_save_system("harmful.xlsx", "harmless.xlsx", mode="image")

        run_training_workflow.assert_called_once_with("harmful.xlsx", "harmless.xlsx", mode="image")
        save_training_artifacts.assert_called_once()
        self.assertEqual(artifact_dir, Path.cwd() / "artifacts" / "image")

    @patch("guardaikids.service.save_training_artifacts")
    @patch("guardaikids.service.run_training_workflow")
    def test_train_and_save_system_passes_image_model_and_xai(self, run_training_workflow, save_training_artifacts):
        run_training_workflow.return_value = {"train_df": [], "val_df": []}

        train_and_save_system(
            "harmful.xlsx",
            "harmless.xlsx",
            mode="text",
            image_analysis_model="clip",
            xai_method="gradient_tokens",
        )

        self.assertEqual(save_training_artifacts.call_args.kwargs["image_analysis_model"], "clip")
        self.assertEqual(save_training_artifacts.call_args.kwargs["xai_method"], "gradient_tokens")

    @patch("guardaikids.service.explain_video")
    @patch("guardaikids.service.predict_video_text")
    @patch("guardaikids.service.extract_thumbnail_features_from_url")
    @patch("guardaikids.service.fetch_youtube_metadata")
    @patch("guardaikids.service.build_youtube_client")
    @patch("guardaikids.service.load_analysis_artifacts")
    def test_analyze_youtube_url_uses_thumbnail_features_for_image_mode(
        self,
        load_artifacts,
        build_client,
        fetch_metadata,
        extract_thumbnail_features,
        predict_video_text,
        explain_video,
    ):
        fake_model = type("FakeModel", (), {"mode": "image"})()
        fake_tokenizer = object()
        load_artifacts.return_value = type(
            "Artifacts",
            (),
            {
                "model": fake_model,
                "tokenizer": fake_tokenizer,
                "metadata": {"mode": "image"},
                "artifact_dir": Path.cwd() / "artifacts" / "image",
            },
        )()
        fetch_metadata.return_value = {
            "video_id": "abc123def45",
            "title": "Demo",
            "description": "Desc",
            "channel": "Channel",
            "published_at": "2026-03-22",
            "thumbnail_url": "https://img.youtube.com/demo.jpg",
            "transcript": "",
        }
        extract_thumbnail_features.return_value = np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)
        predict_video_text.return_value = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        explain_video.return_value = {
            "decision": "Warn",
            "trigger_category": "PH",
            "probability": 0.3,
            "threshold": 0.1,
            "explanation_bullets": ["Decision: Warn."],
            "human_readable_explanation": "Image-based warning.",
            "top_tokens": [],
            "risk_categories": [{"label": "PH", "probability": 0.3}],
            "image_highlights": [{"label": "physical harm or self-injury", "score": 0.7}],
            "modality_summary": "Decision driven primarily by thumbnail evidence.",
        }

        result = analyze_youtube_url("https://www.youtube.com/watch?v=abc123def45", "api-key", mode="image")

        load_artifacts.assert_called_once_with(None, mode="image")
        build_client.assert_called_once_with("api-key")
        extract_thumbnail_features.assert_called_once_with(
            "https://img.youtube.com/demo.jpg",
            image_analysis_model="clip",
        )
        predict_video_text.assert_called_once()
        self.assertEqual(result["analysis_mode"], "image")
        self.assertEqual(result["metadata"]["thumbnail_url"], "https://img.youtube.com/demo.jpg")

    @patch("guardaikids.service.explain_video")
    @patch("guardaikids.service.predict_video_text")
    @patch("guardaikids.service.extract_thumbnail_features_from_url")
    @patch("guardaikids.service.fetch_youtube_metadata")
    @patch("guardaikids.service.build_youtube_client")
    @patch("guardaikids.service.load_analysis_artifacts")
    def test_analyze_youtube_url_passes_image_model_and_xai_overrides(
        self,
        load_artifacts,
        build_client,
        fetch_metadata,
        extract_thumbnail_features,
        predict_video_text,
        explain_video,
    ):
        fake_model = type("FakeModel", (), {"mode": "multimodal"})()
        fake_tokenizer = object()
        load_artifacts.return_value = type(
            "Artifacts",
            (),
            {
                "model": fake_model,
                "tokenizer": fake_tokenizer,
                "metadata": {"mode": "multimodal"},
                "artifact_dir": Path.cwd() / "artifacts" / "multimodal",
            },
        )()
        fetch_metadata.return_value = {
            "video_id": "abc123def45",
            "title": "Demo",
            "description": "Desc",
            "channel": "Channel",
            "published_at": "2026-03-22",
            "thumbnail_url": "https://img.youtube.com/demo.jpg",
            "transcript": "",
        }
        extract_thumbnail_features.return_value = np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)
        predict_video_text.return_value = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        explain_video.return_value = {
            "decision": "Warn",
            "trigger_category": "PH",
            "probability": 0.3,
            "threshold": 0.1,
            "explanation_bullets": ["Decision: Warn."],
            "human_readable_explanation": "Image-based warning.",
            "top_tokens": [],
            "risk_categories": [{"label": "PH", "probability": 0.3}],
            "image_highlights": [{"label": "physical harm or self-injury", "score": 0.7}],
            "modality_summary": "Decision driven by both text and thumbnail evidence.",
        }

        result = analyze_youtube_url(
            "https://www.youtube.com/watch?v=abc123def45",
            "api-key",
            mode="multimodal",
            image_analysis_model="clip",
            xai_method="gradient_tokens",
        )

        extract_thumbnail_features.assert_called_once_with(
            "https://img.youtube.com/demo.jpg",
            image_analysis_model="clip",
        )
        self.assertEqual(explain_video.call_args.kwargs["xai_method"], "gradient_tokens")
        self.assertEqual(result["image_analysis_model"], "clip")
        self.assertEqual(result["xai_method"], "gradient_tokens")


if __name__ == "__main__":
    unittest.main()
