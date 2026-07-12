"""
============================================================
  AI / Machine Learning — Logistic Regression Gradient Descent
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


def np_binary_cross_entropy(labels, probs, eps=1e-12):
    """Return binary cross-entropy from labels and probabilities."""
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))


def np_logistic_loss_and_grads(X, y, w, b):
    """Return (loss, dw, db) for one full-batch logistic regression step."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    logits = X @ w + float(b)
    probs = np_sigmoid(logits)
    loss = np_binary_cross_entropy(y, probs)
    error = probs - y
    n = X.shape[0]
    dw = X.T @ error / n
    db = np.mean(error)
    return loss, dw, db


def np_train_logistic_gd(X, y, lr=0.5, steps=1000):
    """Train logistic regression with vectorized gradient descent."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []

    for _ in range(steps):
        loss, dw, db = np_logistic_loss_and_grads(X, y, w, b)
        w = w - lr * dw
        b = b - lr * db
        losses.append(loss)

    return w, b, losses


def np_predict_proba(X, w, b):
    """Return probabilities from a trained NumPy model."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    return np_sigmoid(X @ w + float(b))


def np_predict_class(X, w, b, threshold=0.5):
    """Return 0/1 class predictions from a trained NumPy model."""
    probs = np_predict_proba(X, w, b)
    return (probs >= threshold).astype(int)


# ======================================================================
# LEVEL 2 — PyTorch
# ======================================================================

def torch_train_logistic_autograd(X, y, lr=0.3, steps=300):
    """Train a torch.nn.Linear logistic regression model."""
    import torch

    X = X.float()
    y = y.float()
    model = torch.nn.Linear(X.shape[1], 1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []

    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    return model, losses


def torch_predict_proba(model, X):
    """Return probabilities from a trained PyTorch model."""
    import torch

    with torch.no_grad():
        logits = model(X.float())
        return torch.sigmoid(logits)


# ======================================================================
# TESTS — do not edit
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


X_TRAIN = np.array([
    [-2.0, -1.0],
    [-1.0, -1.5],
    [-1.0, 0.0],
    [1.0, 0.5],
    [1.5, 1.0],
    [2.0, 1.0],
])
y_TRAIN = np.array([0, 0, 0, 1, 1, 1], dtype=float)


def test_numpy():
    w0 = np.zeros(2)
    loss, dw, db = np_logistic_loss_and_grads(X_TRAIN, y_TRAIN, w0, 0.0)
    check(abs(loss - 0.6931471805599453) < 1e-6, f"initial loss wrong: {loss}")
    check(dw.shape == (2,), f"dw shape wrong: {dw.shape}")
    check(abs(db) < 1e-9, f"db should be near zero for balanced labels: {db}")
    print("PASS  np_logistic_loss_and_grads")

    w, b, losses = np_train_logistic_gd(X_TRAIN, y_TRAIN, lr=0.5, steps=500)
    check(len(losses) == 500, f"loss count wrong: {len(losses)}")
    check(losses[-1] < 0.08, f"final NumPy loss too high: {losses[-1]}")
    print("PASS  np_train_logistic_gd")

    probs = np_predict_proba(X_TRAIN, w, b)
    check(probs[:3].max() < 0.2 and probs[3:].min() > 0.8, f"probs not separated: {probs}")
    print("PASS  np_predict_proba")

    preds = np_predict_class(X_TRAIN, w, b)
    check(np.array_equal(preds, y_TRAIN.astype(int)), f"predictions wrong: {preds}")
    print("PASS  np_predict_class")


def _run_torch_tests():
    import torch

    torch.manual_seed(0)
    X = torch.tensor(X_TRAIN, dtype=torch.float32)
    y = torch.tensor(y_TRAIN, dtype=torch.float32).view(-1, 1)

    model, losses = torch_train_logistic_autograd(X, y, lr=0.3, steps=300)
    check(len(losses) == 300, f"torch loss count wrong: {len(losses)}")
    check(losses[-1] < losses[0] and losses[-1] < 0.15, f"torch training did not converge: {losses[0]} -> {losses[-1]}")
    print("PASS  torch_train_logistic_autograd")

    probs = torch_predict_proba(model, X)
    check(tuple(probs.shape) == (6, 1), f"prob shape wrong: {probs.shape}")
    preds = (probs >= 0.5).long().view(-1)
    check(torch.equal(preds, torch.tensor([0, 0, 0, 1, 1, 1])), f"torch predictions wrong: {preds}")
    print("PASS  torch_predict_proba")


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

