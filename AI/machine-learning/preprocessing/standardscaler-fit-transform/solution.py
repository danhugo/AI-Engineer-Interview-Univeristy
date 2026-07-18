"""Reference solutions for StandardScaler Fit and Transform."""

import numpy as np

def fit_standard_scaler(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return mean, scale

def transform_standard_scaler(X, mean, scale):
    return (np.asarray(X, dtype=float) - mean) / scale

def fit_transform_standard_scaler(X):
    mean, scale = fit_standard_scaler(X)
    return transform_standard_scaler(X, mean, scale), mean, scale

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)
    Z, mean, scale = fit_transform_standard_scaler(X)
    check(np.allclose(mean, [2, 20]), "mean")
    print("PASS  mean")
    check(np.allclose(Z.mean(axis=0), [0, 0]), "scaled mean")
    print("PASS  scaled mean")
    check(np.allclose(Z.std(axis=0), [1, 1]), "scaled std")
    print("PASS  scaled std")
    print("All tests passed.")
