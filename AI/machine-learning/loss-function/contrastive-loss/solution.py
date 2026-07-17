"""Reference solutions for Contrastive Loss."""

import numpy as np


def np_l2_normalize(x, eps=1e-12):
    """Return row-wise L2-normalized array."""
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def np_nt_xent_loss(z1, z2, temperature=0.5):
    """Return SimCLR-style NT-Xent loss."""
    z = np_l2_normalize(np.vstack([z1, z2]))
    n = z1.shape[0]
    sim = (z @ z.T) / temperature
    losses = []
    for i in range(2 * n):
        pos = i + n if i < n else i - n
        logits = np.delete(sim[i], i)
        labels_index = pos if pos < i else pos - 1
        shifted = logits - np.max(logits)
        log_probs = shifted - np.log(np.sum(np.exp(shifted)))
        losses.append(-log_probs[labels_index])
    return float(np.mean(losses))


def torch_nt_xent_loss(z1, z2, temperature=0.5):
    """Return SimCLR-style NT-Xent loss using torch operations."""
    import torch
    z = torch.cat([z1, z2], dim=0)
    z = torch.nn.functional.normalize(z, dim=1)
    n = z1.shape[0]
    sim = (z @ z.T) / temperature
    losses = []
    for i in range(2 * n):
        pos = i + n if i < n else i - n
        mask = torch.ones(2 * n, dtype=torch.bool, device=z.device)
        mask[i] = False
        logits = sim[i][mask].unsqueeze(0)
        target = torch.tensor([pos if pos < i else pos - 1], device=z.device)
        losses.append(torch.nn.functional.cross_entropy(logits, target))
    return torch.mean(torch.stack(losses))


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
