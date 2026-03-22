import unittest
from unittest.mock import patch

from guardaikids import main as main_module


class MainCliTests(unittest.TestCase):
    @patch("guardaikids.main.print_training_summary")
    @patch("guardaikids.main.train_and_save_system")
    def test_train_command_passes_mode_override(self, train_and_save_system, _print_summary):
        train_and_save_system.return_value = ({"train_df": [], "val_df": [], "default_summary": {"classification_report": ""}, "policy_metrics": {}}, None)
        with patch("sys.argv", ["guardaikids", "train", "--mode", "image"]):
            main_module.main()

        self.assertEqual(train_and_save_system.call_args.kwargs["mode"], "image")

    @patch("guardaikids.main.print_training_summary")
    @patch("guardaikids.main.train_and_save_system")
    def test_train_command_passes_image_model_and_xai_overrides(self, train_and_save_system, _print_summary):
        train_and_save_system.return_value = ({"train_df": [], "val_df": [], "default_summary": {"classification_report": ""}, "policy_metrics": {}}, None)
        with patch("sys.argv", ["guardaikids", "train", "--image-model", "clip", "--xai-method", "gradient_tokens"]):
            main_module.main()

        self.assertEqual(train_and_save_system.call_args.kwargs["image_analysis_model"], "clip")
        self.assertEqual(train_and_save_system.call_args.kwargs["xai_method"], "gradient_tokens")

    @patch("guardaikids.main.print_analysis_summary")
    @patch("guardaikids.main.analyze_youtube_url")
    def test_analyze_command_passes_mode_override(self, analyze_youtube_url, _print_summary):
        analyze_youtube_url.return_value = {
            "metadata": {"title": "", "channel": "", "published_at": ""},
            "model_scores": {},
            "recommendations": {"0_4": {"decision": "Allow", "explanation": ""}, "5_8": {"decision": "Allow", "explanation": ""}, "9_12": {"decision": "Allow", "explanation": ""}},
        }
        with patch("sys.argv", ["guardaikids", "analyze", "--url", "https://www.youtube.com/watch?v=abc123def45", "--mode", "multimodal"]):
            with patch.dict("os.environ", {"YOUTUBE_API_KEY": "api-key"}, clear=False):
                main_module.main()

        self.assertEqual(analyze_youtube_url.call_args.kwargs["mode"], "multimodal")

    @patch("guardaikids.main.print_analysis_summary")
    @patch("guardaikids.main.analyze_youtube_url")
    def test_analyze_command_passes_image_model_and_xai_overrides(self, analyze_youtube_url, _print_summary):
        analyze_youtube_url.return_value = {
            "metadata": {"title": "", "channel": "", "published_at": ""},
            "model_scores": {},
            "recommendations": {"0_4": {"decision": "Allow", "explanation": ""}, "5_8": {"decision": "Allow", "explanation": ""}, "9_12": {"decision": "Allow", "explanation": ""}},
        }
        with patch("sys.argv", ["guardaikids", "analyze", "--url", "https://www.youtube.com/watch?v=abc123def45", "--image-model", "clip", "--xai-method", "gradient_tokens"]):
            with patch.dict("os.environ", {"YOUTUBE_API_KEY": "api-key"}, clear=False):
                main_module.main()

        self.assertEqual(analyze_youtube_url.call_args.kwargs["image_analysis_model"], "clip")
        self.assertEqual(analyze_youtube_url.call_args.kwargs["xai_method"], "gradient_tokens")


if __name__ == "__main__":
    unittest.main()
