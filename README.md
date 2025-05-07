# LNN


A Powerful, Modular, and Easy-to-Use Library for Liquid Neural Networks built on PyTorch

🚀 Overview

Liquid Neural Networks (LNNs) are dynamic neural networks that adapt their internal structure in real-time, inspired by neuroscience. They utilize continuous-time differential equations, allowing efficient adaptation and learning, particularly suited for sequential, temporal, and dynamically evolving datasets.

This library provides a robust and modular implementation of Liquid Neural Networks, making them as easy to use as traditional deep learning layers. Whether you're researching dynamic neural architectures or deploying them into production, LNN offers flexible, efficient, and powerful tools.

🎯 Why Use This Library?

Modularity: Easily customizable layers and models.

Flexibility: Numerous configuration options (learnable time constants, solvers, activations).

Ease of Use: Clean and intuitive API, suitable for rapid prototyping.

Performance: Optimized for efficient training and inference.

Integration: Seamlessly integrates into your existing PyTorch workflows.

📂 Repository Structure

lnn/
├── layers/
│   ├── liquid.py           # Core Liquid Neural Network layer
│   ├── neural_ode.py       # Neural ODE integration
│   ├── recurrent_liquid.py # Liquid layer with recurrent capabilities
│   └── utils.py            # Layer-specific utilities
│
├── models/
│   ├── liquid_rnn.py       # Liquid RNN model
│   ├── liquid_classifier.py# Classification model using Liquid layers
│   └── liquid_forecaster.py# Forecasting model using Liquid layers
│
├── utils/
│   ├── activations.py      # Custom activation functions
│   ├── initializations.py  # Weight initialization utilities
│   └── visualization.py    # Tools for visualizing neuron dynamics
│
├── examples/
│   ├── basic_usage.py      # Basic implementation example
│   └── advanced_configuration.py # Advanced setup examples
│
├── tests/
│   ├── test_layers.py      # Unit tests for layers
│   └── test_models.py      # Unit tests for models
│
├── setup.py                # Installation script
├── requirements.txt        # Dependencies
└── README.md               # This document

🛠 Installation

Install via PyPI (coming soon):

pip install lnn

Or directly from GitHub:

git clone https://github.com/yourusername/lnn.git
cd lnn
pip install .

🔥 Quick Start

Here's how easily you can start:

import torch
from lnn.models.liquid_classifier import LiquidClassifier

# Instantiate the model
model = LiquidClassifier(
    input_dim=5,
    hidden_dim=64,
    num_classes=3,
    time_steps=50,
    activation='tanh',
    learn_tau=True,
    solver='rk4'
)

# Dummy input data
inputs = torch.randn(16, 50, 5)  # [Batch, Time, Features]

# Forward pass
outputs = model(inputs)
print(outputs.shape)  # Output shape: [16, 3]

📖 Documentation

Layers Documentation

Models Documentation

Examples and Tutorials

(Documentation pages coming soon!)

🌟 Contributing

We welcome contributions! To contribute, please:

Fork the repository.

Create a new feature branch (git checkout -b feature/amazing-feature).

Commit your changes (git commit -m 'Add amazing feature').

Push to your branch (git push origin feature/amazing-feature).

Open a Pull Request.

📝 Citation

If you find this library helpful in your research, please cite:

@misc{your2025lnn,
  author = {Your Name},
  title = {Liquid Neural Networks (LNN) Library},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/lnn}}
}

📧 Contact

For questions, feature requests, or discussions, reach out:

GitHub Issues: https://github.com/yourusername/lnn/issues

Email: your.email@example.com

📜 License

This project is licensed under the MIT License. See LICENSE for details.

Built with ❤️ for the AI and Machine Learning Community.
