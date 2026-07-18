"""Reference solutions for Linear Kernel."""

import numpy as np

def linear_kernel(X, Z):
    return np.asarray(X, dtype=float) @ np.asarray(Z, dtype=float).T

def linear_decision_function(X, w, b):
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b

def predict_linear_svm(X, w, b):
    return np.where(linear_decision_function(X, w, b) >= 0, 1, -1)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    K = linear_kernel([[1,2],[3,4]], [[1,0],[0,1]])
    check(np.array_equal(K, [[1,2],[3,4]]), "kernel matrix")
    print("PASS  kernel matrix")
    scores = linear_decision_function([[1,1],[-1,-1]], [1,1], 0)
    check(np.array_equal(scores, [2,-2]), "decision scores")
    print("PASS  decision scores")
    check(np.array_equal(predict_linear_svm([[1,1],[-1,-1]], [1,1], 0), [1,-1]), "predictions")
    print("PASS  predictions")
    print("All tests passed.")
