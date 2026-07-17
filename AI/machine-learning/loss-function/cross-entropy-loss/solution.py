"""Reference solutions for Cross-Entropy Loss."""

import numpy as np


def np_stable_softmax(logits):
    """Return row-wise softmax probabilities."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def np_cross_entropy_from_logits(logits, labels):
    """Return mean cross-entropy from raw logits and class ID labels."""
    labels = np.asarray(labels, dtype=int)
    probs = np_stable_softmax(logits)
    true_probs = probs[np.arange(labels.shape[0]), labels]
    return np.mean(-np.log(true_probs))


def torch_cross_entropy_loss(logits, labels):
    """Return PyTorch CrossEntropyLoss."""
    import torch
    return torch.nn.CrossEntropyLoss()(logits, labels.long())


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


LOGITS = np.array([[2.0, 1.0, 0.0],
                   [0.0, 1.0, 2.0]])
LABELS = np.array([0, 2])


def test_numpy():
    probs = np_stable_softmax(LOGITS)
    check(np.allclose(probs.sum(axis=1), [1.0, 1.0]), "softmax rows must sum to 1")
    loss = np_cross_entropy_from_logits(LOGITS, LABELS)
    check(close(loss, 0.40760596444438046), f"cross entropy wrong: {loss}")
    print("PASS  NumPy")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return
    logits = torch.tensor(LOGITS, dtype=torch.float32)
    labels = torch.tensor(LABELS, dtype=torch.long)
    loss = torch_cross_entropy_loss(logits, labels)
    check(close(loss, 0.40760594606399536), f"torch loss wrong: {loss}")
    print("PASS  PyTorch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
