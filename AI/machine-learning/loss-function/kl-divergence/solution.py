"""Reference solutions for KL Divergence."""

import numpy as np


def np_kl_divergence(p, q, eps=1e-12):
    """Return KL(p || q) averaged over rows."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.mean(np.sum(p * np.log(p / q), axis=1))


def torch_kl_divergence(log_q, p):
    """Return PyTorch KLDivLoss with batchmean reduction."""
    import torch
    return torch.nn.KLDivLoss(reduction="batchmean")(log_q, p.float())


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


def test_numpy():
    p = np.array([[0.7, 0.3], [0.5, 0.5]])
    q = np.array([[0.6, 0.4], [0.5, 0.5]])
    loss = np_kl_divergence(p, q)
    check(close(loss, 0.01080043, tol=1e-6), f"np KL wrong: {loss}")
    print("PASS  NumPy")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return
    p = torch.tensor([[0.7, 0.3], [0.5, 0.5]])
    q = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
    log_q = torch.log(q)
    loss = torch_kl_divergence(log_q, p)
    check(close(loss, 0.01080043, tol=1e-6), f"torch KL wrong: {loss}")
    print("PASS  PyTorch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
