"""AI / Machine Learning — Label Smoothing Cross-Entropy Practice."""

import numpy as np


def np_smooth_one_hot(labels, num_classes, epsilon):
    """Return smoothed one-hot labels."""
    pass


def np_label_smoothing_cross_entropy(logits, labels, epsilon=0.1):
    """Return mean label-smoothed cross-entropy from logits."""
    pass


def torch_label_smoothing_cross_entropy(logits, labels, epsilon=0.1):
    """Return PyTorch CrossEntropyLoss with label_smoothing."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


def test_numpy():
    labels = np.array([0, 2])
    smooth = np_smooth_one_hot(labels, 3, 0.1)
    check(np.allclose(smooth[0], [0.9333333333, 0.0333333333, 0.0333333333]), "smooth labels wrong")
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    loss = np_label_smoothing_cross_entropy(logits, labels, 0.1)
    check(close(loss, 0.507606, tol=1e-5), f"np loss wrong: {loss}")
    print("PASS  NumPy")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    labels = torch.tensor([0, 2])
    loss = torch_label_smoothing_cross_entropy(logits, labels, 0.1)
    check(close(loss, 0.507606, tol=1e-5), f"torch loss wrong: {loss}")
    print("PASS  PyTorch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
