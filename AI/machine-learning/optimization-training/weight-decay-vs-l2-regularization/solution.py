"""Reference solutions for Weight Decay vs L2 Regularization."""

import numpy as np


def sgd_with_l2(w, grad, lr, l2_lambda):
    """Return one SGD update when L2 contributes l2_lambda * w to the gradient."""
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    return w - lr * (grad + l2_lambda * w)


def sgd_with_weight_decay(w, grad, lr, weight_decay):
    """Return one SGD update with decoupled weight decay."""
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    return (1 - lr * weight_decay) * w - lr * grad


def adam_like_coupled_l2_step(w, grad, lr, l2_lambda, preconditioner):
    """Return a toy Adam-like update where L2 is added before adaptive scaling."""
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    preconditioner = np.asarray(preconditioner, dtype=float)
    return w - lr * preconditioner * (grad + l2_lambda * w)


def adam_like_decoupled_decay_step(w, grad, lr, weight_decay, preconditioner):
    """Return a toy AdamW-like update where decay is separate from adaptive scaling."""
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    preconditioner = np.asarray(preconditioner, dtype=float)
    return (1 - lr * weight_decay) * w - lr * preconditioner * grad


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_sgd_equivalence():
    w = np.array([2.0, -4.0, 0.5])
    grad = np.array([0.1, -0.2, 0.3])
    lr = 0.05
    lam = 0.1

    l2_step = sgd_with_l2(w, grad, lr, lam)
    decay_step = sgd_with_weight_decay(w, grad, lr, lam)
    check(np.allclose(l2_step, decay_step), "SGD L2 and weight decay should match")
    print("PASS  sgd_equivalence")


def test_adam_like_difference():
    w = np.array([2.0, -4.0])
    grad = np.array([0.1, -0.2])
    preconditioner = np.array([10.0, 0.1])
    lr = 0.01
    lam = 0.1

    coupled = adam_like_coupled_l2_step(w, grad, lr, lam, preconditioner)
    decoupled = adam_like_decoupled_decay_step(w, grad, lr, lam, preconditioner)
    check(not np.allclose(coupled, decoupled), "adaptive scaling should break equivalence")
    check(np.allclose(decoupled, [1.988, -3.9958]), f"decoupled update wrong: {decoupled}")
    print("PASS  adam_like_difference")


if __name__ == "__main__":
    test_sgd_equivalence()
    test_adam_like_difference()
    print("All tests passed.")
