"""Reference solutions for Focal Loss."""

import numpy as np


def np_sigmoid(x):
    """Return sigmoid."""
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))


def np_binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0, eps=1e-12):
    """Return mean binary focal loss from logits."""
    targets = np.asarray(targets, dtype=float)
    p = np_sigmoid(logits)
    p_t = np.where(targets == 1, p, 1 - p)
    alpha_t = np.where(targets == 1, alpha, 1 - alpha)
    return np.mean(-alpha_t * ((1 - p_t) ** gamma) * np.log(np.clip(p_t, eps, 1 - eps)))


def torch_binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Return mean binary focal loss from logits using torch operations."""
    import torch
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p = torch.sigmoid(logits)
    p_t = torch.where(targets == 1, p, 1 - p)
    alpha_t = torch.where(targets == 1, torch.tensor(alpha, device=logits.device), torch.tensor(1 - alpha, device=logits.device))
    return torch.mean(alpha_t * ((1 - p_t) ** gamma) * bce)


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
