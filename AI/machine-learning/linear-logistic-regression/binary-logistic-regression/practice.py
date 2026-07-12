"""
============================================================
  AI / Machine Learning — Binary Logistic Regression
  PRACTICE FILE
  Write every function yourself. Tests tell you if you got
  it right. Do NOT open solution.py first.
============================================================

HOW TO USE
----------
1. Read the hint for each function.
2. Delete the `pass` and write your code.
3. Run:  python practice.py
4. A test PASS means your logic is correct. Fix until all pass.
5. Only open solution.py after you finish or are truly stuck.

Requirements: numpy. PyTorch tests SKIP if torch is missing.
"""

import numpy as np


# ======================================================================
# LEVEL 1 — NumPy
# ======================================================================

def np_sigmoid(logits):
    """
    Return sigmoid(logits).

    HINT:
      Use 1 / (1 + np.exp(-logits)).
    """
    # TODO
    pass


def np_logits(X, w, b):
    """
    Return the raw logistic regression scores: X @ w + b.

    HINT:
      Convert inputs to NumPy arrays with dtype=float.
    """
    # TODO
    pass


def np_predict_proba(X, w, b):
    """
    Return probabilities for class 1.

    HINT:
      1. Compute logits.
      2. Apply np_sigmoid.
    """
    # TODO
    pass


def np_predict_class(X, w, b, threshold=0.5):
    """
    Return class predictions 0 or 1.

    HINT:
      1. Get probabilities.
      2. Return (probs >= threshold).astype(int).
    """
    # TODO
    pass


def np_binary_cross_entropy(labels, probs, eps=1e-12):
    """
    Return binary cross-entropy from labels and probabilities.

    HINT:
      1. Clip probs into [eps, 1 - eps].
      2. Use -mean(y*log(p) + (1-y)*log(1-p)).
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — PyTorch
# ======================================================================

def torch_sigmoid(logits):
    """
    Return torch sigmoid probabilities.

    HINT:
      import torch
      return torch.sigmoid(logits)
    """
    # TODO
    pass


def torch_bce_with_logits_loss(logits, labels):
    """
    Return binary cross-entropy from raw logits.

    HINT:
      Use torch.nn.BCEWithLogitsLoss()(logits, labels.float()).
      Do not apply sigmoid first.
    """
    # TODO
    pass


def torch_predict_class_from_logits(logits, threshold=0.5):
    """
    Return class predictions from raw logits.

    HINT:
      1. Apply torch.sigmoid.
      2. Compare with threshold.
      3. Convert to long/int labels.
    """
    # TODO
    pass


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

