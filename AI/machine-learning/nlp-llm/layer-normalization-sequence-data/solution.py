"""Reference solutions for Layer Normalization for Sequence Data."""

import numpy as np


def layer_norm_stats(X):
    """Return mean and variance over the hidden dimension."""
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    return mean, var


def layer_norm_forward(X, gamma, beta, eps=1e-5):
    """Apply layer normalization over the hidden dimension."""
    X = np.asarray(X, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mean, var = layer_norm_stats(X)
    x_hat = (X - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


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
