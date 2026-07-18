"""Reference solutions for One-Hot Encoding."""

import numpy as np

def fit_one_hot(values):
    return np.array(sorted(np.unique(values)), dtype=object)

def transform_one_hot(values, categories, handle_unknown="error"):
    values = np.asarray(values, dtype=object)
    categories = np.asarray(categories, dtype=object)
    out = np.zeros((len(values), len(categories)), dtype=float)
    lookup = {cat: i for i, cat in enumerate(categories)}
    for r, val in enumerate(values):
        if val not in lookup:
            if handle_unknown == "ignore":
                continue
            raise ValueError(f"unknown category: {val}")
        out[r, lookup[val]] = 1.0
    return out

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    cats = fit_one_hot(["red", "blue", "red"])
    check(cats.tolist() == ["blue", "red"], "categories")
    print("PASS  categories")
    X = transform_one_hot(["red", "blue", "green"], cats, handle_unknown="ignore")
    check(np.array_equal(X, [[0, 1], [1, 0], [0, 0]]), "encoded matrix")
    print("PASS  encoded matrix")
    print("All tests passed.")
