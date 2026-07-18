"""Reference solutions for Polynomial Kernel."""

import numpy as np

def polynomial_kernel(X, Z, degree=3, gamma=1.0, coef0=1.0):
    return (gamma * (np.asarray(X, dtype=float) @ np.asarray(Z, dtype=float).T) + coef0) ** degree

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    K = polynomial_kernel([[1,2]], [[3,4]], degree=2, gamma=1, coef0=1)
    check(np.array_equal(K, [[144]]), "polynomial value")
    print("PASS  polynomial value")
    K2 = polynomial_kernel([[1,0],[0,1]], [[1,0]], degree=2, gamma=1, coef0=0)
    check(np.array_equal(K2, [[1],[0]]), "orthogonal value")
    print("PASS  orthogonal value")
    print("All tests passed.")
