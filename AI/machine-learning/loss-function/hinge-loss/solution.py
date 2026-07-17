"""Reference solutions for Hinge Loss."""

import numpy as np


def np_binary_hinge_loss(scores, labels):
    """Return mean binary hinge loss."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return np.mean(np.maximum(0.0, 1.0 - labels * scores))


def np_multiclass_hinge_loss(scores, labels, margin=1.0):
    """Return mean multi-class hinge loss."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = scores.shape[0]
    true_scores = scores[np.arange(n), labels][:, None]
    losses = np.maximum(0.0, scores - true_scores + margin)
    losses[np.arange(n), labels] = 0.0
    return np.mean(np.sum(losses, axis=1))


def torch_binary_hinge_loss(scores, labels):
    """Return mean binary hinge loss with torch operations."""
    import torch
    return torch.mean(torch.clamp(1.0 - labels.float() * scores, min=0.0))


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


def test_numpy():
    scores = np.array([2.0, 0.2, -0.5])
    labels = np.array([1.0, 1.0, -1.0])
    check(close(np_binary_hinge_loss(scores, labels), (0.0 + 0.8 + 0.5) / 3), "binary hinge wrong")

    class_scores = np.array([[3.0, 1.0, 0.0],
                             [1.0, 2.0, 4.0]])
    class_labels = np.array([0, 1])
    check(close(np_multiclass_hinge_loss(class_scores, class_labels), 1.5), "multiclass hinge wrong")
    print("PASS  NumPy")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return
    scores = torch.tensor([2.0, 0.2, -0.5])
    labels = torch.tensor([1.0, 1.0, -1.0])
    check(close(torch_binary_hinge_loss(scores, labels), (0.0 + 0.8 + 0.5) / 3), "torch hinge wrong")
    print("PASS  PyTorch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
