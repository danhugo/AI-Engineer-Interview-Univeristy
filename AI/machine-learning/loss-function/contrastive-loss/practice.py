"""AI / Machine Learning — Contrastive Loss Practice."""

import numpy as np


def np_l2_normalize(x, eps=1e-12):
    """Return row-wise L2-normalized array."""
    pass


def np_nt_xent_loss(z1, z2, temperature=0.5):
    """
    Return SimCLR-style NT-Xent loss.

    HINT:
      1. Stack z1 and z2 into a 2N batch.
      2. Normalize rows.
      3. Similarity matrix = z @ z.T / temperature.
      4. For each row, positive index is i+N or i-N.
      5. Exclude self-comparison from denominator.
    """
    pass


def torch_nt_xent_loss(z1, z2, temperature=0.5):
    """Return SimCLR-style NT-Xent loss using torch operations."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


Z1 = np.array([[1.0, 0.0], [0.0, 1.0]])
Z2 = np.array([[0.9, 0.1], [0.1, 0.9]])


def test_numpy():
    loss = np_nt_xent_loss(Z1, Z2, temperature=0.5)
    check(close(loss, 0.29646033, tol=1e-5), f"np NT-Xent wrong: {loss}")
    print("PASS  NumPy")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return
    z1 = torch.tensor(Z1, dtype=torch.float32)
    z2 = torch.tensor(Z2, dtype=torch.float32)
    loss = torch_nt_xent_loss(z1, z2, temperature=0.5)
    check(close(loss, 0.29646033, tol=1e-5), f"torch NT-Xent wrong: {loss}")
    print("PASS  PyTorch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
