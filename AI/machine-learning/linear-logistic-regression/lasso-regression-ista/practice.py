"""
============================================================
  AI / Machine Learning — Lasso Regression with ISTA
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

def np_soft_threshold(x, threshold):
    """
    Apply soft-thresholding elementwise.

    HINT:
      1. Convert x to a NumPy array.
      2. Use np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0).
    """
    # TODO
    pass


def np_lasso_predict(X, w, b):
    """
    Predict with a linear model.

    HINT:
      Return X @ w + b.
    """
    # TODO
    pass


def np_lasso_loss(X, y, w, b, alpha):
    """
    Return MSE + alpha * L1 penalty.
    Do not regularize the bias b.

    HINT:
      1. pred = np_lasso_predict(X, w, b)
      2. mse = np.mean((pred - y) ** 2)
      3. penalty = alpha * np.sum(np.abs(w))
    """
    # TODO
    pass


def np_lasso_ista(X, y, alpha=0.1, lr=0.05, steps=1000):
    """
    Fit Lasso regression with ISTA.
    Return (w, b).

    HINT:
      1. Start w as zeros and b as 0.0.
      2. Compute MSE gradients for w and b.
      3. Take a gradient step.
      4. Soft-threshold only w with threshold lr * alpha.
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — PyTorch
# ======================================================================

def torch_soft_threshold(x, threshold):
    """
    Apply soft-thresholding to a torch tensor.

    HINT:
      Use torch.sign, torch.abs, and torch.clamp(..., min=0.0).
    """
    # TODO
    pass


def torch_lasso_loss(X, y, w, b, alpha):
    """
    Return MSE + alpha * L1 penalty using PyTorch tensors.
    Do not regularize the bias b.
    """
    # TODO
    pass


def torch_lasso_ista(X, y, alpha=0.1, lr=0.05, steps=1000):
    """
    Fit Lasso regression with tensor ISTA.
    Return (w, b).

    HINT:
      This is the same loop as NumPy, but with torch operations.
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
