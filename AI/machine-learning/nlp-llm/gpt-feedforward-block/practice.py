"""
AI / Machine Learning — GPT Feed-Forward Block
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""

import numpy as np


def gelu(x):
    """
    Return tanh-approximate GELU.

    Formula:
      0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    pass


def feedforward_forward(X, W1, b1, W2, b2):
    """
    Apply a position-wise GPT-style feed-forward block.

    X shape: (batch, sequence, hidden)
    W1 shape: (hidden, inner)
    W2 shape: (inner, hidden)
    """
    pass


def feedforward_parameter_count(hidden_size, inner_size, bias=True):
    """Return trainable parameter count for two dense projections."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_gelu_known_values():
    x = np.array([-1.0, 0.0, 1.0])
    out = gelu(x)
    expected = np.array([-0.15880801, 0.0, 0.84119199])
    check(np.allclose(out, expected), f"gelu wrong: {out}")
    print("PASS  gelu_known_values")


def test_feedforward_forward_shape_and_values():
    X = np.array([[[1.0, -1.0], [0.5, 2.0]]])
    W1 = np.array([[1.0, 0.0, 2.0], [0.5, -1.0, 1.0]])
    b1 = np.array([0.0, 1.0, -0.5])
    W2 = np.array([[1.0, -1.0], [0.0, 2.0], [0.5, 0.5]])
    b2 = np.array([0.1, -0.2])
    out = feedforward_forward(X, W1, b1, W2, b2)
    expected = np.array([[[0.61857101, 3.53633838], [2.74202944, -0.67472973]]])
    check(out.shape == X.shape, f"shape wrong: {out.shape}")
    check(np.allclose(out, expected), f"forward wrong: {out}")
    print("PASS  feedforward_forward_shape_and_values")


def test_parameter_count():
    check(feedforward_parameter_count(768, 3072) == 4_722_432, "count with bias wrong")
    check(feedforward_parameter_count(10, 40, bias=False) == 800, "count without bias wrong")
    print("PASS  parameter_count")


if __name__ == "__main__":
    test_gelu_known_values()
    test_feedforward_forward_shape_and_values()
    test_parameter_count()
    print("All tests passed.")
