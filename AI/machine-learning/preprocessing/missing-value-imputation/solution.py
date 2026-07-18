"""Reference solutions for Missing Value Imputation."""

import numpy as np

def fit_numeric_imputer(X, strategy="mean"):
    X = np.asarray(X, dtype=float)
    if strategy == "mean":
        return np.nanmean(X, axis=0)
    if strategy == "median":
        return np.nanmedian(X, axis=0)
    raise ValueError("strategy must be mean or median")

def transform_numeric_imputer(X, fill_values):
    X = np.asarray(X, dtype=float).copy()
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = np.asarray(fill_values, dtype=float)[cols]
    return X

def fit_mode(values):
    vals = [v for v in values if v is not None]
    uniq, counts = np.unique(vals, return_counts=True)
    return uniq[np.argmax(counts)]

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]])
    fill = fit_numeric_imputer(X, "mean")
    check(np.allclose(fill, [2.0, 6.0]), "mean fill")
    print("PASS  mean fill")
    Xt = transform_numeric_imputer(X, fill)
    check(not np.isnan(Xt).any(), "no missing")
    print("PASS  no missing")
    check(fit_mode(["a", "b", "a", None]) == "a", "mode")
    print("PASS  mode")
    print("All tests passed.")
