"""
AI / Machine Learning — Layer Normalization for Sequence Data
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""

import numpy as np


def layer_norm_stats(X):
    """
    Return mean and variance for layer norm on sequence data.

    X has shape (batch, sequence, hidden).

    HINT:
      Normalize over the last axis and keep dimensions for broadcasting.
    """
    pass


def layer_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Apply layer normalization over the hidden dimension.

    HINT:
      x_hat = (X - mean) / sqrt(var + eps)
      out = gamma * x_hat + beta
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_layer_norm_stats_shape_and_values():
    X = np.array([
        [[1.0, 2.0, 3.0], [10.0, 12.0, 14.0]],
        [[2.0, 4.0, 6.0], [1.0, 1.0, 1.0]],
    ])
    mean, var = layer_norm_stats(X)
    check(mean.shape == (2, 2, 1), f"mean shape wrong: {mean.shape}")
    check(var.shape == (2, 2, 1), f"var shape wrong: {var.shape}")
    check(np.allclose(mean[..., 0], [[2.0, 12.0], [4.0, 1.0]]), f"mean wrong: {mean}")
    check(np.allclose(var[..., 0], [[2.0 / 3.0, 8.0 / 3.0], [8.0 / 3.0, 0.0]]), f"var wrong: {var}")
    print("PASS  layer_norm_stats_shape_and_values")


def test_layer_norm_forward():
    X = np.array([[[1.0, 2.0, 3.0]]])
    gamma = np.array([1.0, 2.0, 0.5])
    beta = np.array([0.0, 1.0, -1.0])
    out = layer_norm_forward(X, gamma, beta, eps=0.0)
    expected = np.array([[[-1.22474487, 1.0, -0.38762756]]])
    check(np.allclose(out, expected), f"forward wrong: {out}")
    print("PASS  layer_norm_forward")


def test_independent_tokens():
    X = np.array([[[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]]])
    out = layer_norm_forward(X, gamma=np.ones(3), beta=np.zeros(3), eps=0.0)
    check(np.allclose(out[0, 0], out[0, 1]), f"tokens should normalize to same pattern: {out}")
    print("PASS  independent_tokens")


if __name__ == "__main__":
    test_layer_norm_stats_shape_and_values()
    test_layer_norm_forward()
    test_independent_tokens()
    print("All tests passed.")
