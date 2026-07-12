"""
============================================================
  AI / Machine Learning — Binary Logistic Regression
  SOLUTION FILE  (reference)
  Drop-in replacement for practice.py — same signatures.
============================================================

Requirements: numpy. PyTorch tests SKIP if torch is missing.
"""

import numpy as np


# ======================================================================
# LEVEL 1 — NumPy
# ======================================================================

def np_sigmoid(logits):
    """Return sigmoid(logits)."""
    logits = np.asarray(logits, dtype=float)
    return 1 / (1 + np.exp(-logits))


def np_logits(X, w, b):
    """Return the raw logistic regression scores: X @ w + b."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    return X @ w + float(b)


def np_predict_proba(X, w, b):
    """Return probabilities for class 1."""
    return np_sigmoid(np_logits(X, w, b))


def np_predict_class(X, w, b, threshold=0.5):
    """Return class predictions 0 or 1."""
    probs = np_predict_proba(X, w, b)
    return (probs >= threshold).astype(int)


def np_binary_cross_entropy(labels, probs, eps=1e-12):
    """Return binary cross-entropy from labels and probabilities."""
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))


# ======================================================================
# LEVEL 2 — PyTorch
# ======================================================================

def torch_sigmoid(logits):
    """Return torch sigmoid probabilities."""
    import torch
    return torch.sigmoid(logits)


def torch_bce_with_logits_loss(logits, labels):
    """Return binary cross-entropy from raw logits."""
    import torch
    return torch.nn.BCEWithLogitsLoss()(logits, labels.float())


def torch_predict_class_from_logits(logits, threshold=0.5):
    """Return class predictions from raw logits."""
    import torch
    probs = torch.sigmoid(logits)
    return (probs >= threshold).long()


# ======================================================================
# TESTS — do not edit
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_numpy():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    w = np.array([2.0, -1.0])
    b = -0.5

    logits = np_logits(X, w, b)
    check(np.allclose(logits, [-0.5, 1.5, -1.5, 1.5]), f"np_logits wrong: {logits}")
    print("PASS  np_logits")

    probs = np_sigmoid(np.array([-100.0, 0.0, 100.0]))
    check(np.allclose(probs[1], 0.5), f"np_sigmoid(0) wrong: {probs[1]}")
    check(probs[0] < 1e-40 and probs[2] > 1 - 1e-12, f"np_sigmoid extremes wrong: {probs}")
    print("PASS  np_sigmoid")

    pred_probs = np_predict_proba(X, w, b)
    check(np.allclose(pred_probs, 1 / (1 + np.exp(-logits))), f"np_predict_proba wrong: {pred_probs}")
    print("PASS  np_predict_proba")

    classes = np_predict_class(X, w, b)
    check(np.array_equal(classes, [0, 1, 0, 1]), f"np_predict_class wrong: {classes}")
    print("PASS  np_predict_class")

    labels = np.array([0, 1, 0, 1])
    loss = np_binary_cross_entropy(labels, pred_probs)
    check(loss < 0.35, f"np_binary_cross_entropy too high: {loss}")
    print("PASS  np_binary_cross_entropy")


def _run_torch_tests():
    import torch

    logits = torch.tensor([[-2.0], [0.0], [2.0]])
    labels = torch.tensor([[0.0], [0.0], [1.0]])

    probs = torch_sigmoid(logits)
    check(torch.allclose(probs[1], torch.tensor([0.5])), f"torch_sigmoid wrong: {probs}")
    print("PASS  torch_sigmoid")

    loss = torch_bce_with_logits_loss(logits, labels)
    check(loss.ndim == 0 and float(loss) < 0.35, f"torch_bce_with_logits_loss wrong: {loss}")
    print("PASS  torch_bce_with_logits_loss")

    classes = torch_predict_class_from_logits(logits)
    check(torch.equal(classes.view(-1), torch.tensor([0, 1, 1])), f"torch_predict_class_from_logits wrong: {classes}")
    print("PASS  torch_predict_class_from_logits")


def run_torch_tests():
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"SKIP  PyTorch tests ({e})")
        return
    _run_torch_tests()


if __name__ == "__main__":
    print("\n── NumPy ──")
    test_numpy()
    print("\n── PyTorch ──")
    run_torch_tests()
    print("\nAll tests passed.")

