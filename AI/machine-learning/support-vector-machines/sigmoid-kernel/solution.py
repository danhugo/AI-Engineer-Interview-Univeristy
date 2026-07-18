"""Reference solutions for Sigmoid Kernel."""

import numpy as np

def sigmoid_kernel(X, Z, gamma=1.0, coef0=0.0):
    return np.tanh(gamma * (np.asarray(X, dtype=float) @ np.asarray(Z, dtype=float).T) + coef0)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    K = sigmoid_kernel([[1,0]], [[1,0],[0,1]], gamma=1, coef0=0)
    check(np.allclose(K, [[np.tanh(1), 0.0]]), "sigmoid values")
    print("PASS  sigmoid values")
    check(np.allclose(sigmoid_kernel([[10]], [[10]]), 1.0), "saturates")
    print("PASS  bounded")
    print("All tests passed.")
