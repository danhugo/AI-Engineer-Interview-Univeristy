"""
============================================================
  AI / Machine Learning — Elastic Net Gradient Descent
  PRACTICE FILE
  Write every function yourself. Tests tell you if you got
  it right. Do NOT open solution.py first.
============================================================

HOW TO USE
----------
1. Read the hint for each function.
2. Delete the `pass` and write your code.
3. Run:  python practice.py
4. A test PASS means your logic is correct. Fix until all pass.
5. Only open solution.py after you finish or are truly stuck.

Requirements:  pip install numpy torch
   (If torch is missing, the PyTorch tests print SKIP instead
    of failing — but install it to really practice PyTorch.)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — NumPy
# ======================================================================

def np_elastic_net_predict(X, w, b):
    """
    Predict with a linear model.

    HINT:
      Return X @ w + b.
    """
    # TODO
    pass


def np_elastic_net_loss(X, y, w, b, l1=0.1, l2=0.1):
    """
    Return MSE + L1 penalty + L2 penalty.
    Do not regularize the bias b.

    HINT:
      penalty = l1 * np.sum(np.abs(w)) + l2 * np.sum(w ** 2)
    """
    # TODO
    pass


def np_elastic_net_gradients(X, y, w, b, l1=0.1, l2=0.1):
    """
    Return (grad_w, grad_b) for the Elastic Net objective.
    Do not regularize the bias b.

    HINT:
      1. residual = X @ w + b - y
      2. mse_grad_w = (2 / n) * X.T @ residual
      3. grad_w = mse_grad_w + l1 * np.sign(w) + 2 * l2 * w
      4. grad_b = (2 / n) * np.sum(residual)
    """
    # TODO
    pass


def np_fit_elastic_net_gd(X, y, l1=0.1, l2=0.1, lr=0.01, steps=3000):
    """
    Fit Elastic Net regression with gradient descent.
    Return (w, b).

    HINT:
      Start w and b at zero, compute gradients, then update.
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — PyTorch
# ======================================================================

def torch_elastic_net_loss(X, y, w, b, l1=0.1, l2=0.1):
    """
    Return Elastic Net loss using PyTorch tensors.
    Do not regularize the bias b.
    """
    # TODO
    pass


def torch_elastic_net_gradients(X, y, w, b, l1=0.1, l2=0.1):
    """
    Return (grad_w, grad_b) using PyTorch tensor operations.

    HINT:
      Use torch.sign for the L1 subgradient.
    """
    # TODO
    pass


def torch_fit_elastic_net_gd(X, y, l1=0.1, l2=0.1, lr=0.01, steps=3000):
    """
    Fit Elastic Net regression with manual tensor gradient descent.
    Return (w, b).
    """
    # TODO
    pass


# ======================================================================
# TESTS — do not edit
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

