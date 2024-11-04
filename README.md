# MLCookieCutter

MLCookieCutter is an open-source collection of customizable, ready-to-use machine learning model templates. This project aims to provide easy-to-follow templates for popular architectures (CNNs, RNNs, Transformers, etc.) that are adaptable to various tasks and datasets. The goal is to make it easier for ML engineers and learners to quickly prototype and experiment with different models without building them from scratch.

## Features
- **Model Templates**: Includes templates for CNNs, RNNs, Transformers, and more.
- **Customizable**: Easily adjust hyperparameters, layers, and other configurations to fit your use case.
- **Educational**: Provides well-documented code and examples to help you understand each architecture.

## Getting Started

### Prerequisites
- Python 3.12 or higher
- Poetry for dependency management (install with `pip install poetry`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MLCookieCutter.git
   cd MLCookieCutter

2. Install dependencies:
    ```bash
    poetry install
    ```
## Usage
To use any model template, you can import it directly from the mlcookiecutter.templates module. Here’s an example with the SimpleCNN template:

```python
from mlcookiecutter.templates.cnn_template import SimpleCNN

# Initialize the model
model = SimpleCNN(num_classes=10)

# Example forward pass
import torch
sample_input = torch.randn(1, 3, 32, 32)
output = model(sample_input)
print(output)
```

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING](./CONTRIBUTING.md) file for guidelines on how to contribute.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.