import unittest
from unittest.mock import patch

from guardaikids.youtube import build_model_input, build_youtube_client, extract_video_id


class YouTubeHelperTests(unittest.TestCase):
    def test_extract_video_id_supports_standard_watch_urls(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=2dDpryw3z5w"), "2dDpryw3z5w")

    def test_build_model_input_concatenates_metadata_fields(self):
        metadata = {"title": "T", "description": "D", "transcript": "X"}
        self.assertEqual(build_model_input(metadata), "T D X")

    def test_build_youtube_client_surfaces_optional_dependency_message(self):
        with patch("guardaikids.youtube._import_youtube_dependencies", side_effect=ImportError("missing optional deps")):
            with self.assertRaisesRegex(ImportError, "missing optional deps"):
                build_youtube_client("demo-key")


if __name__ == "__main__":
    unittest.main()
