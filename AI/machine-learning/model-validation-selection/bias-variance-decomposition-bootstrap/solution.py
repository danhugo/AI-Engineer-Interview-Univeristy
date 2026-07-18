"""Reference solutions for Bias-Variance Decomposition from Bootstrap."""

import numpy as np

def bias_variance_from_predictions(predictions, y_true):
    predictions = np.asarray(predictions, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mean_pred = predictions.mean(axis=0)
    bias2_per_point = (mean_pred - y_true) ** 2
    variance_per_point = ((predictions - mean_pred) ** 2).mean(axis=0)
    return float(bias2_per_point.mean()), float(variance_per_point.mean())

def bootstrap_sample_indices(n_samples, seed=None):
    return np.random.default_rng(seed).integers(0, n_samples, size=n_samples)

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    preds = np.array([[1.0, 2.0], [1.5, 1.5], [0.5, 2.5]])
    bias2, var = bias_variance_from_predictions(preds, [1.0, 2.0])
    check(np.isclose(bias2, 0.0), "zero average bias")
    print("PASS  zero average bias")
    check(np.isclose(var, 1/6), "variance estimate")
    print("PASS  variance estimate")
    idx = bootstrap_sample_indices(4, seed=0)
    check(len(idx) == 4 and idx.max() < 4, "bootstrap indices")
    print("PASS  bootstrap indices")
    print("All tests passed.")
