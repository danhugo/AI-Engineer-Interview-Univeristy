"""Reference solutions for Huber Loss."""

import numpy as np


def np_huber_loss(y_true, y_pred, delta=1.0):
    """Return mean Huber loss."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)
    return np.mean(np.where(abs_error <= delta, quadratic, linear))


def torch_huber_loss(y_true, y_pred, delta=1.0):
    """Return PyTorch HuberLoss."""
    import torch
    return torch.nn.HuberLoss(delta=delta)(y_pred, y_true.float())


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


def test_numpy():
    y = np.array([0.0, 0.0, 0.0])
    pred = np.array([0.5, 1.0, 3.0])
    expected = (0.125 + 0.5 + 2.5) / 3
    check(close(np_huber_loss(y, pred, delta=1.0), expected), "np_huber_loss wrong")
    print("PASS  NumPy")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return
    y = torch.tensor([0.0, 0.0, 0.0])
    pred = torch.tensor([0.5, 1.0, 3.0])
    expected = (0.125 + 0.5 + 2.5) / 3
    check(close(torch_huber_loss(y, pred, delta=1.0), expected), "torch_huber_loss wrong")
    print("PASS  PyTorch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
