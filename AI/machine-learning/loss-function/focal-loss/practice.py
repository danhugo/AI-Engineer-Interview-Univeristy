"""AI / Machine Learning — Focal Loss Practice."""

import numpy as np


def np_sigmoid(x):
    """Return sigmoid."""
    pass


def np_binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0, eps=1e-12):
    """
    Return mean binary focal loss from logits.

    HINT:
      p = sigmoid(logits)
      p_t = p for target 1, else 1-p
      alpha_t = alpha for target 1, else 1-alpha
    """
    pass


def torch_binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Return mean binary focal loss from logits using torch operations."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


def test_numpy():
    logits = np.array([2.0, -1.0, 0.0])
    targets = np.array([1.0, 0.0, 1.0])
    loss = np_binary_focal_loss(logits, targets)
    check(close(loss, 0.02025538, tol=1e-5), f"np focal wrong: {loss}")
    print("PASS  NumPy")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return
    logits = torch.tensor([2.0, -1.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = torch_binary_focal_loss(logits, targets)
    check(close(loss, 0.02025538, tol=1e-5), f"torch focal wrong: {loss}")
    print("PASS  PyTorch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
