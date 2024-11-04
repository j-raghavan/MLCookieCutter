import unittest
import torch
from mlcookiecutter.templates.cnn_template import SimpleCNN

class TestSimpleCNN(unittest.TestCase):
    def test_initialization(self):
        model = SimpleCNN(num_classes=10)
        self.assertIsInstance(model, SimpleCNN)

    def test_forward_pass(self):
        model = SimpleCNN(num_classes=10)
        sample_input = torch.randn(1, 3, 32, 32)
        output = model(sample_input)
        self.assertEqual(output.shape, (1, 10))

if __name__ == '__main__':
    unittest.main()
