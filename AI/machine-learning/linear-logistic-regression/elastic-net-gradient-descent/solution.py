"""
Reference solutions for AI / Machine Learning — Elastic Net Gradient Descent.

Open this only after attempting practice.py.
"""

import numpy as np


def np_elastic_net_predict(X, w, b):
    """Predict with a linear model."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    return X @ w + b


def np_elastic_net_loss(X, y, w, b, l1=0.1, l2=0.1):
    """Return MSE + L1 penalty + L2 penalty. Do not regularize b."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    pred = np_elastic_net_predict(X, w, b)
    mse = np.mean((pred - y) ** 2)
    penalty = l1 * np.sum(np.abs(w)) + l2 * np.sum(w ** 2)
    return mse + penalty


def np_elastic_net_gradients(X, y, w, b, l1=0.1, l2=0.1):
    """Return (grad_w, grad_b) for the Elastic Net objective."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = X.shape[0]
    residual = X @ w + b - y
    mse_grad_w = (2 / n) * X.T @ residual
    grad_w = mse_grad_w + l1 * np.sign(w) + 2 * l2 * w
    grad_b = (2 / n) * np.sum(residual)
    return grad_w, grad_b


def np_fit_elastic_net_gd(X, y, l1=0.1, l2=0.1, lr=0.01, steps=3000):
    """Fit Elastic Net regression with gradient descent. Return (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0

    for _ in range(steps):
        grad_w, grad_b = np_elastic_net_gradients(X, y, w, b, l1, l2)
        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b


def torch_elastic_net_loss(X, y, w, b, l1=0.1, l2=0.1):
    """Return Elastic Net loss using PyTorch tensors."""
    import torch

    pred = X @ w + b
    mse = torch.mean((pred - y) ** 2)
    penalty = l1 * torch.sum(torch.abs(w)) + l2 * torch.sum(w ** 2)
    return mse + penalty


def torch_elastic_net_gradients(X, y, w, b, l1=0.1, l2=0.1):
    """Return (grad_w, grad_b) using PyTorch tensor operations."""
    import torch

    n = X.shape[0]
    residual = X @ w + b - y
    mse_grad_w = (2 / n) * X.T @ residual
    grad_w = mse_grad_w + l1 * torch.sign(w) + 2 * l2 * w
    grad_b = (2 / n) * torch.sum(residual)
    return grad_w, grad_b


def torch_fit_elastic_net_gd(X, y, l1=0.1, l2=0.1, lr=0.01, steps=3000):
    """Fit Elastic Net regression with manual tensor gradient descent."""
    import torch

    X = X.float()
    y = y.float()
    w = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
    b = torch.tensor(0.0, dtype=X.dtype, device=X.device)

    for _ in range(steps):
        grad_w, grad_b = torch_elastic_net_gradients(X, y, w, b, l1, l2)
        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b


# ======================================================================
# TESTS — kept identical to practice.py
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


X_DATA = np.array([[0.0, 0.0],
                   [1.0, 0.0],
                   [2.0, 1.0],
                   [3.0, 1.0],
                   [4.0, 2.0],
                   [5.0, 2.0]])
y_DATA = 3.0 * X_DATA[:, 0] - 1.0 * X_DATA[:, 1] + 2.0


def test_numpy():
    w = np.array([3.0, -1.0])
    b = 2.0
    pred = np_elastic_net_predict(X_DATA, w, b)
    check(np.allclose(pred, y_DATA), f"prediction wrong: {pred}")
    print("PASS  np_elastic_net_predict")

    loss = np_elastic_net_loss(X_DATA, y_DATA, w, b, l1=0.5, l2=0.25)
    check(close(loss, 4.5), f"loss should be only penalties for exact fit: {loss}")
    print("PASS  np_elastic_net_loss")

    grad_w, grad_b = np_elastic_net_gradients(X_DATA, y_DATA, w, b, l1=0.5, l2=0.25)
    check(np.allclose(grad_w, [2.0, -1.0]), f"gradient should be penalty-only: {grad_w}")
    check(close(grad_b, 0.0), f"bias gradient should be zero: {grad_b}")
    print("PASS  np_elastic_net_gradients")

    w_fit, b_fit = np_fit_elastic_net_gd(X_DATA, y_DATA, l1=0.01, l2=0.01, lr=0.01, steps=8000)
    trained_pred = np_elastic_net_predict(X_DATA, w_fit, b_fit)
    check(np.mean((trained_pred - y_DATA) ** 2) < 0.05, f"fit has high MSE: {trained_pred}")
    check(abs(b_fit - 2.0) < 0.35, f"bias should be close and not regularized: {b_fit}")
    print("PASS  np_fit_elastic_net_gd")


def _run_torch_tests():
    import torch

    X = torch.tensor(X_DATA, dtype=torch.float32)
    y = torch.tensor(y_DATA, dtype=torch.float32)
    w = torch.tensor([3.0, -1.0])
    b = torch.tensor(2.0)

    loss = torch_elastic_net_loss(X, y, w, b, l1=0.5, l2=0.25)
    check(close(loss, 4.5), f"torch loss wrong: {loss}")
    print("PASS  torch_elastic_net_loss")

    grad_w, grad_b = torch_elastic_net_gradients(X, y, w, b, l1=0.5, l2=0.25)
    check(torch.allclose(grad_w, torch.tensor([2.0, -1.0])), f"torch gradient wrong: {grad_w}")
    check(close(grad_b, 0.0), f"torch bias gradient wrong: {grad_b}")
    print("PASS  torch_elastic_net_gradients")

    w_fit, b_fit = torch_fit_elastic_net_gd(X, y, l1=0.01, l2=0.01, lr=0.01, steps=8000)
    pred = X @ w_fit + b_fit
    check(float(torch.mean((pred - y) ** 2)) < 0.05, f"torch fit has high MSE: {pred}")
    check(abs(float(b_fit) - 2.0) < 0.35, f"torch bias should be close: {b_fit}")
    print("PASS  torch_fit_elastic_net_gd")


def run_torch_tests():
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"SKIP  PyTorch tests ({e})")
        return
    _run_torch_tests()


if __name__ == "__main__":
    test_numpy()
    run_torch_tests()
    print("\nAll tests passed.")
