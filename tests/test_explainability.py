import unittest
from unittest.mock import patch

import torch

from guardaikids.explainability import clean_tokens, explain_text_tokens, explain_video, get_input_embeddings


class DummyModel:
    def __init__(self):
        self.embedding = torch.nn.Embedding(8, 4)
        self.mode = "text"

    @property
    def device(self):
        return torch.device("cpu")

    def eval(self):
        return self

    def zero_grad(self):
        return None

    def __call__(self, inputs_embeds=None, attention_mask=None, image_features=None):
        logits = inputs_embeds.mean(dim=1)[:, :4]
        return type("Output", (), {"logits": logits})()

    def get_input_embeddings(self):
        return self.embedding


class DummyMultimodalModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.mode = "multimodal"

    @property
    def device(self):
        return torch.device("cpu")

    def eval(self):
        return self

    def zero_grad(self):
        return None

    def __call__(self, inputs_embeds=None, attention_mask=None, image_features=None):
        if image_features is None:
            raise ValueError("image_features are required when mode='multimodal'.")
        logits = inputs_embeds.mean(dim=1)[:, :4] + image_features[:, :4]
        return type("Output", (), {"logits": logits})()


class DummyTokenizer:
    def __call__(self, text, return_tensors=None, truncation=None, padding=None, max_length=None):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        return ["<s>", "alpha", "beta", "gamma"]


class ExplainabilityTests(unittest.TestCase):
    def test_clean_tokens_handles_sentencepiece_and_wordpiece_prefixes(self):
        tokens = ["\u2581hello", "world", "##wide"]
        self.assertEqual(clean_tokens(tokens), ["hello", "worldwide"])

    def test_get_input_embeddings_uses_model_api(self):
        model = DummyModel()
        self.assertIs(get_input_embeddings(model), model.embedding)

    def test_explain_text_tokens_accepts_image_features_for_multimodal_mode(self):
        model = DummyMultimodalModel()
        tokenizer = DummyTokenizer()

        top_tokens = explain_text_tokens(
            "demo text",
            category_index=0,
            model=model,
            tokenizer=tokenizer,
            image_features=[0.1] * 526,
        )

        self.assertIsInstance(top_tokens, list)

    @patch("guardaikids.explainability.explain_text_tokens", return_value=[("adult", 0.9), ("movie", 0.7)])
    def test_explain_video_returns_bullet_payload_with_multiple_risk_categories(self, _mock_explain_tokens):
        model = DummyModel()
        tokenizer = DummyTokenizer()

        explanation = explain_video(
            text="adult movie with violence",
            probs_row=[0.05, 0.65, 0.55, 0.10],
            age_group="0_4",
            model=model,
            tokenizer=tokenizer,
            image_features=[0.0] * 526,
        )

        self.assertIn("explanation_bullets", explanation)
        self.assertGreaterEqual(len(explanation["risk_categories"]), 2)
        self.assertTrue(any("Highest detected risk categories" in bullet for bullet in explanation["explanation_bullets"]))

    @patch("guardaikids.explainability.explain_text_tokens", return_value=[("ending", 0.9), ("horror", 0.7)])
    def test_explain_video_hides_low_confidence_allow_cues(self, _mock_explain_tokens):
        model = DummyModel()
        tokenizer = DummyTokenizer()

        explanation = explain_video(
            text="ending gameplay horror",
            probs_row=[0.05, 0.04, 0.06, 0.08],
            age_group="0_4",
            model=model,
            tokenizer=tokenizer,
            image_features=[0.0] * 526,
        )

        self.assertEqual(explanation["decision"], "Allow")
        self.assertEqual(explanation["top_tokens"], [])
        self.assertEqual(explanation["image_highlights"], [])
        self.assertTrue(
            any("No text or thumbnail cue exceeded the current warning threshold." in bullet for bullet in explanation["explanation_bullets"])
        )


if __name__ == "__main__":
    unittest.main()
