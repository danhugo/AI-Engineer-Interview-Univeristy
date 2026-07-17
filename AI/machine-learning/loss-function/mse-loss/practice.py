"""
AI / Machine Learning — MSE Loss
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""

import numpy as np


def np_mse(y_true, y_pred):
    """
    Return mean squared error.

    HINT:
      Convert inputs to float arrays and use np.mean((y_pred - y_true) ** 2).
    """
    pass


def np_mse_grad(y_true, y_pred):
    """
    Return gradient of MSE with respect to y_pred.

    HINT:
      For mean MSE, gradient is (2 / n) * (y_pred - y_true).
    """
    pass


def torch_mse_loss(y_true, y_pred):
    """
    Return PyTorch MSE loss.

    HINT:
      Use torch.nn.MSELoss()(y_pred, y_true.float()).
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


def test_numpy():
    y = np.array([1.0, 2.0, 4.0])
    pred = np.array([1.0, 3.0, 2.0])
    check(close(np_mse(y, pred), 5.0 / 3.0), "np_mse wrong")
    check(np.allclose(np_mse_grad(y, pred), [0.0, 2.0 / 3.0, -4.0 / 3.0]), "np_mse_grad wrong")
    print("PASS  NumPy")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return
    y = torch.tensor([1.0, 2.0, 4.0])
    pred = torch.tensor([1.0, 3.0, 2.0])
    check(close(torch_mse_loss(y, pred), 5.0 / 3.0), "torch_mse_loss wrong")
    print("PASS  PyTorch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
