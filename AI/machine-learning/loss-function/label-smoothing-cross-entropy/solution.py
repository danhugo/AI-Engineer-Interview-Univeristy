"""Reference solutions for Label Smoothing Cross-Entropy."""

import numpy as np


def np_smooth_one_hot(labels, num_classes, epsilon):
    """Return smoothed one-hot labels."""
    labels = np.asarray(labels, dtype=int)
    out = np.full((labels.shape[0], num_classes), epsilon / num_classes, dtype=float)
    out[np.arange(labels.shape[0]), labels] += 1.0 - epsilon
    return out


def np_label_smoothing_cross_entropy(logits, labels, epsilon=0.1):
    """Return mean label-smoothed cross-entropy from logits."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    targets = np_smooth_one_hot(labels, logits.shape[1], epsilon)
    return np.mean(-np.sum(targets * log_probs, axis=1))


def torch_label_smoothing_cross_entropy(logits, labels, epsilon=0.1):
    """Return PyTorch CrossEntropyLoss with label_smoothing."""
    import torch
    return torch.nn.CrossEntropyLoss(label_smoothing=epsilon)(logits, labels.long())


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
