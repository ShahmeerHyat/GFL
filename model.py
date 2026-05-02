"""
model.py
========

Minimal MLP backbone for tabular intrusion detection.

The architecture is deliberately small and standard:
    input -> Linear(256) -> BN -> ReLU -> Dropout -> Linear(128) -> BN -> ReLU -> Dropout -> Linear(K)

Holding the architecture fixed across all loss-function comparisons is a
deliberate design choice: the contribution is the loss, not the network.
"""

from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Tabular-data MLP classifier.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_classes : int
        Number of output classes.
    hidden_dims : sequence of int
        Hidden layer widths.
    dropout : float
        Dropout probability applied after each hidden block.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# Tests   —   run with: python model.py
# ===========================================================================
def _run_tests() -> None:
    torch.manual_seed(0)

    # 1. Forward pass returns expected shape.
    m = MLP(input_dim=78, num_classes=15)
    x = torch.randn(32, 78)
    y = m(x)
    assert y.shape == (32, 15), f"unexpected output shape: {tuple(y.shape)}"
    print(f"  [pass] forward shape ({tuple(y.shape)})")

    # 2. Backward pass produces finite gradients.
    loss = y.sum()
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
    print(f"  [pass] backward produces finite gradients")

    # 3. Model size is sensible (a few hundred K parameters).
    n = m.num_parameters()
    assert 10_000 < n < 1_000_000, f"unexpected parameter count: {n}"
    print(f"  [pass] {n:,} parameters")

    # 4. Eval mode disables dropout / freezes BN running stats.
    m.eval()
    y1 = m(x)
    y2 = m(x)
    assert torch.allclose(y1, y2), "eval mode should be deterministic but isn't"
    print(f"  [pass] eval mode deterministic")

    print("\nAll model tests passed.")


if __name__ == "__main__":
    print("Running model.py self-tests...\n")
    _run_tests()
