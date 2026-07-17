"""
AI / Machine Learning — Batch Normalization Forward Pass
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""

import numpy as np


def batch_mean_var(X):
    """
    Return per-feature batch mean and variance for a 2D array.

    HINT:
      Use axis=0 and population variance, not sample variance.
    """
    pass


def batch_norm_forward_train(X, gamma, beta, eps=1e-5):
    """
    Return batch-normalized output using current batch statistics.

    HINT:
      x_hat = (X - mean) / np.sqrt(var + eps)
      out = gamma * x_hat + beta
    """
    pass


def update_running_stats(running_mean, running_var, batch_mean, batch_var, momentum):
    """
    Return updated running mean and variance.

    HINT:
      new = momentum * old + (1 - momentum) * batch_stat
    """
    pass


def batch_norm_forward_eval(X, gamma, beta, running_mean, running_var, eps=1e-5):
    """
    Return batch-normalized output using running statistics.
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_batch_mean_var():
    X = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 16.0]])
    mean, var = batch_mean_var(X)
    check(np.allclose(mean, [3.0, 40.0 / 3.0]), f"mean wrong: {mean}")
    check(np.allclose(var, [8.0 / 3.0, 56.0 / 9.0]), f"var wrong: {var}")
    print("PASS  batch_mean_var")


def test_train_forward():
    X = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 16.0]])
    gamma = np.array([2.0, 0.5])
    beta = np.array([1.0, -1.0])
    out = batch_norm_forward_train(X, gamma, beta, eps=0.0)
    expected = np.array([
        [-1.44948974, -1.6681531],
        [1.0, -0.86636938],
        [3.44948974, -0.46547752],
    ])
    check(np.allclose(out, expected), f"train forward wrong: {out}")
    print("PASS  train_forward")


def test_running_stats_and_eval():
    running_mean = np.array([0.0, 10.0])
    running_var = np.array([1.0, 4.0])
    batch_mean = np.array([2.0, 14.0])
    batch_var = np.array([9.0, 16.0])
    new_mean, new_var = update_running_stats(
        running_mean, running_var, batch_mean, batch_var, momentum=0.9
    )
    check(np.allclose(new_mean, [0.2, 10.4]), f"running mean wrong: {new_mean}")
    check(np.allclose(new_var, [1.8, 5.2]), f"running var wrong: {new_var}")

    X = np.array([[2.0, 14.0]])
    out = batch_norm_forward_eval(X, gamma=np.ones(2), beta=np.zeros(2),
                                  running_mean=batch_mean, running_var=batch_var, eps=0.0)
    check(np.allclose(out, [0.0, 0.0]), f"eval forward wrong: {out}")
    print("PASS  running_stats_and_eval")


if __name__ == "__main__":
    test_batch_mean_var()
    test_train_forward()
    test_running_stats_and_eval()
    print("All tests passed.")
