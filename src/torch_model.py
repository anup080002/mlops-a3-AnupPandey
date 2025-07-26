"""
Reusable lightweight PyTorch definition that mirrors the scikit‑learn
linear‑regression model (one fully connected layer, no activation).
"""

import torch.nn as nn


class SingleLayer(nn.Module):
    """y = W·x + b  (no bias if you prefer intercept_=False)."""

    def __init__(self, in_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=bias)

    def forward(self, x):
        return self.linear(x)
