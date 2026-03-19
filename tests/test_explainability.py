import unittest

import torch

from etp.explainability import clean_tokens, get_input_embeddings


class DummyModel:
    def __init__(self):
        self.embedding = torch.nn.Embedding(8, 4)

    def get_input_embeddings(self):
        return self.embedding


class ExplainabilityTests(unittest.TestCase):
    def test_clean_tokens_handles_sentencepiece_and_wordpiece_prefixes(self):
        tokens = ["\u2581hello", "world", "##wide"]
        self.assertEqual(clean_tokens(tokens), ["hello", "worldwide"])

    def test_get_input_embeddings_uses_model_api(self):
        model = DummyModel()
        self.assertIs(get_input_embeddings(model), model.embedding)


if __name__ == "__main__":
    unittest.main()
