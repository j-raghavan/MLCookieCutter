import unittest

import torch

from mlcookiecutter.templates.cnn_template import SimpleCNN
from mlcookiecutter.templates.rnn_template import SimpleRNN
from mlcookiecutter.templates.transformer_template import SimpleTransformer


class TestTemplates(unittest.TestCase):
    def test_cnn_template(self):
        model = SimpleCNN(num_classes=10)
        sample_input = torch.randn(1, 3, 32, 32)
        output = model(sample_input)
        self.assertEqual(output.shape, (1, 10))

    def test_rnn_template(self):
        model = SimpleRNN(input_size=28, hidden_size=64, num_layers=1, num_classes=10)
        sample_input = torch.randn(1, 28, 28)  # (batch, seq, input_size)
        output = model(sample_input)
        self.assertEqual(output.shape, (1, 10))

    def test_transformer_template(self):
        model = SimpleTransformer(
            input_dim=1000, num_heads=2, hidden_dim=64, num_layers=1, num_classes=10
        )
        sample_input = torch.randint(0, 1000, (1, 32))  # (batch, seq_len)
        output = model(sample_input)
        self.assertEqual(output.shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
