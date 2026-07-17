"""Reference solutions for Shortcut Connection Gradient Effect."""

import numpy as np


def residual_forward(x, W):
    """Compute y = x + x @ W for a linear residual branch."""
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    return x + x @ W


def plain_forward(x, W):
    """Compute y = x @ W for a plain linear branch."""
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    return x @ W


def residual_input_gradient(upstream_grad, W):
    """Return dL/dx for y = x + x @ W."""
    upstream_grad = np.asarray(upstream_grad, dtype=float)
    W = np.asarray(W, dtype=float)
    identity = np.eye(W.shape[0])
    return upstream_grad @ (identity + W.T)


def plain_input_gradient(upstream_grad, W):
    """Return dL/dx for y = x @ W."""
    upstream_grad = np.asarray(upstream_grad, dtype=float)
    W = np.asarray(W, dtype=float)
    return upstream_grad @ W.T


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_forward_values():
    x = np.array([[1.0, 2.0]])
    W = np.array([[0.1, 0.2], [0.3, 0.4]])
    check(np.allclose(plain_forward(x, W), [[0.7, 1.0]]), "plain forward wrong")
    check(np.allclose(residual_forward(x, W), [[1.7, 3.0]]), "residual forward wrong")
    print("PASS  forward_values")


def test_input_gradients():
    upstream = np.array([[2.0, -1.0]])
    W = np.array([[0.1, 0.2], [0.3, 0.4]])
    plain_grad = plain_input_gradient(upstream, W)
    residual_grad = residual_input_gradient(upstream, W)
    check(np.allclose(plain_grad, [[0.0, 0.2]]), f"plain gradient wrong: {plain_grad}")
    check(np.allclose(residual_grad, [[2.0, -0.8]]), f"residual gradient wrong: {residual_grad}")
    print("PASS  input_gradients")


def test_identity_path_when_branch_zero():
    upstream = np.array([[3.0, -4.0]])
    W = np.zeros((2, 2))
    check(np.allclose(plain_input_gradient(upstream, W), [[0.0, 0.0]]), "plain zero branch gradient wrong")
    check(np.allclose(residual_input_gradient(upstream, W), upstream), "residual identity gradient wrong")
    print("PASS  identity_path_when_branch_zero")


if __name__ == "__main__":
    test_forward_values()
    test_input_gradients()
    test_identity_path_when_branch_zero()
    print("All tests passed.")
