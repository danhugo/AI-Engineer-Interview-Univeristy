"""Reference solutions for MinMaxScaler."""

import numpy as np

def fit_minmax_scaler(X):
    X = np.asarray(X, dtype=float)
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    denom = np.where(data_max - data_min == 0, 1.0, data_max - data_min)
    return data_min, data_max, denom

def transform_minmax_scaler(X, data_min, denom, feature_range=(0.0, 1.0)):
    a, b = feature_range
    scaled = (np.asarray(X, dtype=float) - data_min) / denom
    return scaled * (b - a) + a

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    X = np.array([[1, 10], [3, 20], [5, 30]], dtype=float)
    mn, mx, denom = fit_minmax_scaler(X)
    check(np.allclose(mn, [1, 10]) and np.allclose(mx, [5, 30]), "min max")
    print("PASS  min max")
    Z = transform_minmax_scaler(X, mn, denom)
    check(np.allclose(Z, [[0, 0], [.5, .5], [1, 1]]), "scaled values")
    print("PASS  scaled values")
    print("All tests passed.")
