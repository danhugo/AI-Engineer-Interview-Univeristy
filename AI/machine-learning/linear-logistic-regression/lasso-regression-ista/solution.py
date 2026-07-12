"""
Reference solutions for AI / Machine Learning — Lasso Regression with ISTA.

Open this only after attempting practice.py.
"""

import numpy as np


def np_soft_threshold(x, threshold):
    """Apply soft-thresholding elementwise."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def np_lasso_predict(X, w, b):
    """Predict with a linear model."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    return X @ w + b


def np_lasso_loss(X, y, w, b, alpha):
    """Return MSE + alpha * L1 penalty. Do not regularize b."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    pred = np_lasso_predict(X, w, b)
    mse = np.mean((pred - y) ** 2)
    penalty = alpha * np.sum(np.abs(w))
    return mse + penalty


def np_lasso_ista(X, y, alpha=0.1, lr=0.05, steps=1000):
    """Fit Lasso regression with ISTA. Return (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    n = X.shape[0]

    for _ in range(steps):
        pred = X @ w + b
        residual = pred - y
        grad_w = (2 / n) * X.T @ residual
        grad_b = (2 / n) * np.sum(residual)
        w = np_soft_threshold(w - lr * grad_w, lr * alpha)
        b = b - lr * grad_b

    return w, b


def torch_soft_threshold(x, threshold):
    """Apply soft-thresholding to a torch tensor."""
    import torch

    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)


def torch_lasso_loss(X, y, w, b, alpha):
    """Return MSE + alpha * L1 penalty using PyTorch tensors."""
    import torch

    pred = X @ w + b
    mse = torch.mean((pred - y) ** 2)
    penalty = alpha * torch.sum(torch.abs(w))
    return mse + penalty


def torch_lasso_ista(X, y, alpha=0.1, lr=0.05, steps=1000):
    """Fit Lasso regression with tensor ISTA. Return (w, b)."""
    import torch

    X = X.float()
    y = y.float()
    w = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
    b = torch.tensor(0.0, dtype=X.dtype, device=X.device)
    n = X.shape[0]

    for _ in range(steps):
        pred = X @ w + b
        residual = pred - y
        grad_w = (2 / n) * X.T @ residual
        grad_b = (2 / n) * torch.sum(residual)
        w = torch_soft_threshold(w - lr * grad_w, lr * alpha)
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


X_DATA = np.array([[0.0, 1.0, -2.0],
                   [1.0, -1.0, 1.0],
                   [2.0, 1.0, 0.0],
                   [3.0, -1.0, 2.0],
                   [4.0, 1.0, -1.0],
                   [5.0, -1.0, 3.0]])
y_DATA = 2.0 * X_DATA[:, 0] + 1.0


def test_numpy():
    x = np.array([-2.0, -0.2, 0.0, 0.3, 3.0])
    out = np_soft_threshold(x, 0.5)
    check(np.allclose(out, [-1.5, 0.0, 0.0, 0.0, 2.5]), f"soft threshold wrong: {out}")
    print("PASS  np_soft_threshold")

    pred = np_lasso_predict(X_DATA, np.array([2.0, 0.0, 0.0]), 1.0)
    check(np.allclose(pred, y_DATA), f"prediction wrong: {pred}")
    print("PASS  np_lasso_predict")

    loss = np_lasso_loss(X_DATA, y_DATA, np.array([2.0, 0.0, 0.0]), 1.0, alpha=0.25)
    check(close(loss, 0.5), f"loss should include L1 penalty only: {loss}")
    print("PASS  np_lasso_loss")

    w, b = np_lasso_ista(X_DATA, y_DATA, alpha=0.05, lr=0.01, steps=8000)
    trained_pred = np_lasso_predict(X_DATA, w, b)
    check(np.mean((trained_pred - y_DATA) ** 2) < 0.05, f"ISTA fit has high MSE: {trained_pred}")
    check(abs(w[1]) < 0.05, f"irrelevant feature weight should be near zero: {w}")
    check(abs(w[2]) < 0.05, f"irrelevant feature weight should be near zero: {w}")
    print("PASS  np_lasso_ista")


def _run_torch_tests():
    import torch

    x = torch.tensor([-2.0, -0.2, 0.0, 0.3, 3.0])
    out = torch_soft_threshold(x, 0.5)
    check(torch.allclose(out, torch.tensor([-1.5, 0.0, 0.0, 0.0, 2.5])), f"torch soft threshold wrong: {out}")
    print("PASS  torch_soft_threshold")

    X = torch.tensor(X_DATA, dtype=torch.float32)
    y = torch.tensor(y_DATA, dtype=torch.float32)
    w = torch.tensor([2.0, 0.0, 0.0])
    b = torch.tensor(1.0)
    loss = torch_lasso_loss(X, y, w, b, alpha=0.25)
    check(close(loss, 0.5), f"torch lasso loss wrong: {loss}")
    print("PASS  torch_lasso_loss")

    w_fit, b_fit = torch_lasso_ista(X, y, alpha=0.05, lr=0.01, steps=8000)
    pred = X @ w_fit + b_fit
    check(float(torch.mean((pred - y) ** 2)) < 0.05, f"torch ISTA fit has high MSE: {pred}")
    check(abs(float(w_fit[1])) < 0.05, f"irrelevant feature weight should be near zero: {w_fit}")
    print("PASS  torch_lasso_ista")


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
