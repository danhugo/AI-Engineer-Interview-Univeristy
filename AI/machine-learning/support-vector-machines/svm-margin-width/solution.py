"""Reference solutions for SVM Margin Width."""

import numpy as np

def hyperplane_distance(X, w, b):
    return np.abs(np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b) / np.linalg.norm(w)

def margin_width(w):
    return float(2.0 / np.linalg.norm(w))

def functional_margin(X, y, w, b):
    return np.asarray(y, dtype=float) * (np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    check(np.isclose(margin_width([3,4]), 0.4), "margin width")
    print("PASS  margin width")
    d = hyperplane_distance([[3,0]], [1,0], 0)
    check(np.isclose(d[0], 3.0), "point distance")
    print("PASS  point distance")
    fm = functional_margin([[1,0],[-1,0]], [1,-1], [1,0], 0)
    check(np.array_equal(fm, [1,1]), "functional margin")
    print("PASS  functional margin")
    print("All tests passed.")
