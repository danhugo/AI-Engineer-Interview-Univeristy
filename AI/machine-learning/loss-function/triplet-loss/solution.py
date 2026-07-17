"""Reference solutions for Triplet Loss."""

import numpy as np


def np_pairwise_l2(a, b, eps=1e-12):
    """Return row-wise Euclidean distance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.sum((a - b) ** 2, axis=1) + eps)


def np_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """Return mean triplet margin loss."""
    d_pos = np_pairwise_l2(anchor, positive)
    d_neg = np_pairwise_l2(anchor, negative)
    return np.mean(np.maximum(0.0, d_pos - d_neg + margin))


def torch_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """Return PyTorch TripletMarginLoss."""
    import torch
    return torch.nn.TripletMarginLoss(margin=margin, p=2)(anchor, positive, negative)


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


A = np.array([[0.0, 0.0], [1.0, 1.0]])
P = np.array([[0.0, 0.5], [1.0, 1.2]])
N = np.array([[2.0, 0.0], [1.1, 1.1]])


def test_numpy():
    loss = np_triplet_margin_loss(A, P, N, margin=1.0)
    check(close(loss, 0.52928932, tol=1e-6), f"np triplet wrong: {loss}")
    print("PASS  NumPy")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return
    a = torch.tensor(A, dtype=torch.float32)
    p = torch.tensor(P, dtype=torch.float32)
    n = torch.tensor(N, dtype=torch.float32)
    loss = torch_triplet_margin_loss(a, p, n, margin=1.0)
    check(close(loss, 0.52928932, tol=1e-5), f"torch triplet wrong: {loss}")
    print("PASS  PyTorch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
