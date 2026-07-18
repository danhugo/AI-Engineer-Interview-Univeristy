"""Reference solutions for RBF Gaussian Kernel."""

import numpy as np

def squared_euclidean_distances(X, Z):
    X = np.asarray(X, dtype=float); Z = np.asarray(Z, dtype=float)
    return ((X[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)

def rbf_kernel(X, Z, gamma):
    return np.exp(-gamma * squared_euclidean_distances(X, Z))

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    D = squared_euclidean_distances([[0,0],[1,0]], [[0,0]])
    check(np.array_equal(D, [[0],[1]]), "squared distances")
    print("PASS  squared distances")
    K = rbf_kernel([[0,0],[1,0]], [[0,0]], gamma=1.0)
    check(np.allclose(K, [[1.0],[np.exp(-1)]]), "rbf values")
    print("PASS  rbf values")
    print("All tests passed.")
