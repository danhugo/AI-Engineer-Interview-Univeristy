"""
============================================================
  AI / Machine Learning — Softmax Regression Gradient Descent
  PRACTICE FILE
  Write every function yourself. Tests tell you if you got
  it right. Do NOT open solution.py first.
============================================================

Requirements: numpy. PyTorch tests SKIP if torch is missing.
"""

import numpy as np


# ======================================================================
# LEVEL 1 — NumPy
# ======================================================================

def np_one_hot(labels, num_classes):
    """
    Convert class ID labels into one-hot rows.

    HINT:
      Use np.eye(num_classes)[labels].
    """
    # TODO
    pass


def np_stable_softmax(logits):
    """
    Return row-wise stable softmax probabilities.

    HINT:
      1. Subtract np.max(logits, axis=1, keepdims=True).
      2. Exp the shifted logits.
      3. Divide by row sums.
    """
    # TODO
    pass


def np_cross_entropy_from_probs(labels, probs, eps=1e-12):
    """
    Return mean cross-entropy from class IDs and probabilities.

    HINT:
      1. Pick true class probabilities with probs[np.arange(n), labels].
      2. Clip them before np.log.
      3. Return -mean(log(true_probs)).
    """
    # TODO
    pass


def np_softmax_loss_and_grads(X, labels, W, b):
    """
    Return (loss, dW, db) for one full-batch softmax regression step.

    HINT:
      1. logits = X @ W + b
      2. probs = np_stable_softmax(logits)
      3. one_hot = np_one_hot(labels, number_of_classes)
      4. error = probs - one_hot
      5. dW = X.T @ error / n
      6. db = mean(error, axis=0)
    """
    # TODO
    pass


def np_train_softmax_gd(X, labels, num_classes, lr=0.5, steps=800):
    """
    Train softmax regression with vectorized gradient descent.
    Return (W, b, losses).

    HINT:
      Start W as zeros with shape (num_features, num_classes).
      Start b as zeros with shape (num_classes,).
    """
    # TODO
    pass


def np_predict_class(X, W, b):
    """
    Return predicted class IDs.

    HINT:
      Compute logits, then softmax, then np.argmax(axis=1).
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — PyTorch
# ======================================================================

def torch_cross_entropy_loss(logits, labels):
    """
    Return PyTorch multi-class cross-entropy from raw logits.

    HINT:
      Use torch.nn.CrossEntropyLoss()(logits, labels.long()).
    """
    # TODO
    pass


def torch_train_softmax_autograd(X, labels, num_classes, lr=0.3, steps=300):
    """
    Train a torch.nn.Linear softmax regression model.
    Return (model, losses).

    HINT:
      1. model = torch.nn.Linear(num_features, num_classes)
      2. loss_fn = torch.nn.CrossEntropyLoss()
      3. labels should be long class IDs.
      4. Pass raw logits into the loss.
    """
    # TODO
    pass


def torch_predict_class(model, X):
    """
    Return predicted class IDs from a trained PyTorch model.

    HINT:
      Use torch.no_grad(), model(X.float()), and torch.argmax(axis=1).
    """
    # TODO
    pass


# ======================================================================
# TESTS — do not edit
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


X_TRAIN = np.array([
    [2.0, 0.0],
    [1.5, 0.5],
    [0.0, 2.0],
    [0.5, 1.5],
    [-2.0, -1.0],
    [-1.5, -1.5],
])
y_TRAIN = np.array([0, 0, 1, 1, 2, 2])


def test_numpy():
    oh = np_one_hot(np.array([0, 2, 1]), 3)
    check(np.array_equal(oh, [[1, 0, 0], [0, 0, 1], [0, 1, 0]]), f"one_hot wrong: {oh}")
    print("PASS  np_one_hot")

    logits = np.array([[1000.0, 1001.0, 999.0], [1.0, 1.0, 1.0]])
    probs = np_stable_softmax(logits)
    check(np.allclose(probs.sum(axis=1), [1.0, 1.0]), f"softmax rows should sum to 1: {probs}")
    check(np.argmax(probs[0]) == 1, f"largest logit should win: {probs[0]}")
    print("PASS  np_stable_softmax")

    ce = np_cross_entropy_from_probs(np.array([1, 0]), np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]))
    check(ce < 0.3, f"cross entropy too high: {ce}")
    print("PASS  np_cross_entropy_from_probs")

    W0 = np.zeros((2, 3))
    b0 = np.zeros(3)
    loss, dW, db = np_softmax_loss_and_grads(X_TRAIN, y_TRAIN, W0, b0)
    check(abs(loss - np.log(3)) < 1e-6, f"initial loss wrong: {loss}")
    check(dW.shape == (2, 3), f"dW shape wrong: {dW.shape}")
    check(db.shape == (3,), f"db shape wrong: {db.shape}")
    print("PASS  np_softmax_loss_and_grads")

    W, b, losses = np_train_softmax_gd(X_TRAIN, y_TRAIN, num_classes=3, lr=0.5, steps=800)
    check(len(losses) == 800, f"loss count wrong: {len(losses)}")
    check(losses[-1] < 0.12, f"final NumPy loss too high: {losses[-1]}")
    print("PASS  np_train_softmax_gd")

    preds = np_predict_class(X_TRAIN, W, b)
    check(np.array_equal(preds, y_TRAIN), f"NumPy predictions wrong: {preds}")
    print("PASS  np_predict_class")


def _run_torch_tests():
    import torch

    sample_logits = torch.tensor([[4.0, 1.0, 0.0], [0.0, 2.0, 5.0]])
    sample_labels = torch.tensor([0, 2])
    loss = torch_cross_entropy_loss(sample_logits, sample_labels)
    check(float(loss) < 0.08, f"torch_cross_entropy_loss too high: {loss}")
    print("PASS  torch_cross_entropy_loss")

    torch.manual_seed(0)
    X = torch.tensor(X_TRAIN, dtype=torch.float32)
    y = torch.tensor(y_TRAIN, dtype=torch.long)
    model, losses = torch_train_softmax_autograd(X, y, num_classes=3, lr=0.3, steps=300)
    check(len(losses) == 300, f"torch loss count wrong: {len(losses)}")
    check(losses[-1] < losses[0] and losses[-1] < 0.2, f"torch training did not converge: {losses[0]} -> {losses[-1]}")
    print("PASS  torch_train_softmax_autograd")

    preds = torch_predict_class(model, X)
    check(torch.equal(preds, y), f"torch predictions wrong: {preds}")
    print("PASS  torch_predict_class")


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

