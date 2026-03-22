import unittest
from pathlib import Path
import shutil
from unittest.mock import patch

import torch

from guardaikids.config import IMAGE_FEATURE_DIM
from guardaikids.modeling import (
    CUSTOM_CONFIG_NAME,
    CUSTOM_WEIGHTS_NAME,
    MultimodalSequenceClassifier,
    predict_video_text,
)


class DummyTextEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = torch.nn.Embedding(32, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        return type("Output", (), {"last_hidden_state": inputs_embeds})()

    def get_input_embeddings(self):
        return self.embedding

    def save_pretrained(self, *_args, **_kwargs):
        return None


class DummyAutoConfig:
    hidden_size = 8


class ModelingTests(unittest.TestCase):
    @patch("guardaikids.modeling.AutoConfig.from_pretrained", return_value=DummyAutoConfig())
    @patch("guardaikids.modeling.AutoModel.from_pretrained", return_value=DummyTextEncoder())
    def test_text_mode_forward_returns_logits_and_loss(self, *_mocks):
        model = MultimodalSequenceClassifier(mode="text", num_labels=4)
        outputs = model(
            input_ids=torch.ones((2, 4), dtype=torch.long),
            attention_mask=torch.ones((2, 4), dtype=torch.long),
            labels=torch.ones((2, 4), dtype=torch.float32),
        )
        self.assertEqual(outputs.logits.shape, (2, 4))
        self.assertIsNotNone(outputs.loss)

    @patch("guardaikids.modeling.AutoConfig.from_pretrained", return_value=DummyAutoConfig())
    def test_image_mode_forward_uses_image_features_only(self, *_mocks):
        model = MultimodalSequenceClassifier(mode="image", num_labels=4)
        outputs = model(
            image_features=torch.ones((3, IMAGE_FEATURE_DIM), dtype=torch.float32),
            labels=torch.zeros((3, 4), dtype=torch.float32),
        )
        self.assertEqual(outputs.logits.shape, (3, 4))
        self.assertIsNotNone(outputs.loss)

    @patch("guardaikids.modeling.AutoConfig.from_pretrained", return_value=DummyAutoConfig())
    @patch("guardaikids.modeling.AutoModel.from_pretrained", return_value=DummyTextEncoder())
    def test_multimodal_forward_combines_text_and_image_inputs(self, *_mocks):
        model = MultimodalSequenceClassifier(mode="multimodal", num_labels=4)
        outputs = model(
            input_ids=torch.ones((2, 5), dtype=torch.long),
            attention_mask=torch.ones((2, 5), dtype=torch.long),
            image_features=torch.ones((2, IMAGE_FEATURE_DIM), dtype=torch.float32),
            labels=torch.zeros((2, 4), dtype=torch.float32),
        )
        self.assertEqual(outputs.logits.shape, (2, 4))
        self.assertIsNotNone(outputs.loss)

    @patch("guardaikids.modeling.AutoConfig.from_pretrained", return_value=DummyAutoConfig())
    @patch("guardaikids.modeling.AutoModel.from_pretrained", return_value=DummyTextEncoder())
    def test_from_pretrained_loads_legacy_text_artifact_with_old_image_head(self, *_mocks):
        model_dir = Path("artifacts") / "test_legacy_text_artifact"
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            (model_dir / "text_encoder").mkdir(parents=True, exist_ok=True)
            (model_dir / CUSTOM_CONFIG_NAME).write_text(
                json_dumps(
                    {
                        "model_name": "distilroberta-base",
                        "mode": "text",
                        "num_labels": 4,
                        "image_feature_dim": 518,
                        "fusion_hidden_dim": 8,
                    }
                ),
                encoding="utf-8",
            )
            legacy_state = {
                "text_classifier.weight": torch.zeros((4, 8)),
                "text_classifier.bias": torch.zeros(4),
                "image_classifier.weight": torch.zeros((4, 518)),
                "image_classifier.bias": torch.zeros(4),
                "fusion_layer.0.weight": torch.zeros((8, 526)),
                "fusion_layer.0.bias": torch.zeros(8),
                "fusion_layer.3.weight": torch.zeros((4, 8)),
                "fusion_layer.3.bias": torch.zeros(4),
            }
            torch.save(legacy_state, model_dir / CUSTOM_WEIGHTS_NAME)

            model = MultimodalSequenceClassifier.from_pretrained(model_dir)

            self.assertEqual(model.mode, "text")
        finally:
            shutil.rmtree(model_dir, ignore_errors=True)

    @patch("guardaikids.modeling.AutoConfig.from_pretrained", return_value=DummyAutoConfig())
    def test_predict_video_text_surfaces_image_feature_dim_mismatch(self, *_mocks):
        model = MultimodalSequenceClassifier(mode="image", num_labels=4)

        with self.assertRaisesRegex(ValueError, "expects image feature vectors of length"):
            predict_video_text(model, tokenizer=None, text="", image_features=[0.0] * 10)


def json_dumps(payload):
    import json

    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    unittest.main()
