"""Reference solutions for GPT Feed-Forward Block."""

import numpy as np


def gelu(x):
    """Return tanh-approximate GELU."""
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def feedforward_forward(X, W1, b1, W2, b2):
    """Apply a position-wise GPT-style feed-forward block."""
    X = np.asarray(X, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    hidden = gelu(X @ W1 + b1)
    return hidden @ W2 + b2


def feedforward_parameter_count(hidden_size, inner_size, bias=True):
    """Return trainable parameter count for two dense projections."""
    count = hidden_size * inner_size + inner_size * hidden_size
    if bias:
        count += inner_size + hidden_size
    return count


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
