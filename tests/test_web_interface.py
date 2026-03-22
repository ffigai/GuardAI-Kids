import unittest
from unittest.mock import patch

from guardaikids.web_interface import app


class WebInterfaceTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_get_index_shows_mode_selector(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('name="mode"', html)
        self.assertIn("multimodal", html)

    @patch("guardaikids.web_interface.analyze_youtube_url")
    def test_post_passes_selected_mode(self, analyze_youtube_url):
        analyze_youtube_url.return_value = {
            "analysis_mode": "image",
            "metadata": {
                "title": "Demo",
                "channel": "Channel",
                "published_at": "2026-03-22",
                "thumbnail_url": "https://img.youtube.com/demo.jpg",
            },
            "recommendations": {
                "0_4": {
                    "decision": "Warn",
                    "explanation_bullets": ["Decision: Warn for ages 0-4.", "Thumbnail cues: sexual or explicit content (0.82)."],
                    "top_tokens": [],
                    "image_highlights": [{"label": "sexual or explicit content", "score": 0.82}],
                }
            },
            "model_scores": {},
        }
        with patch.dict("os.environ", {"YOUTUBE_API_KEY": "api-key"}, clear=False):
            response = self.client.post(
                "/",
                data={"url": "https://www.youtube.com/watch?v=abc123def45", "mode": "image"},
            )

        self.assertEqual(response.status_code, 200)
        analyze_youtube_url.assert_called_once_with(
            "https://www.youtube.com/watch?v=abc123def45",
            "api-key",
            mode="image",
        )
        html = response.get_data(as_text=True)
        self.assertIn("Demo", html)
        self.assertIn("Channel | 2026-03-22", html)
        self.assertIn("Decision Guide", html)
        self.assertIn("Basic View", html)
        self.assertIn("Warn", html)
        self.assertIn("sexual or explicit content (82%)", html)


if __name__ == "__main__":
    unittest.main()
