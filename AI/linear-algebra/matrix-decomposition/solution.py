"""
============================================================
  AI / Linear Algebra — Matrix Decomposition (LU, QR, Cholesky)
  SOLUTION FILE  (reference)
  Drop-in replacement for practice.py — same signatures.
  Read this only after you have tried practice.py yourself.
============================================================

Requirements:  pip install numpy torch

Levels:
  LEVEL 1 — Pure Python   (LU + Cholesky from scratch)
  LEVEL 2 — NumPy         (np.linalg.qr, np.linalg.cholesky)
  LEVEL 3 — PyTorch       (torch.linalg.qr, torch.linalg.cholesky)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch)
# ======================================================================

def py_lu(A):
    """LU by Gaussian elimination, no pivoting. Returns (L, U), L@U == A."""
    n = len(A)
    U = [row[:] for row in A]                                  # copy A
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for k in range(n):                                         # pivot column
        for i in range(k + 1, n):                             # rows below pivot
            m = U[i][k] / U[k][k]                              # multiplier
            L[i][k] = m
            for j in range(k, n):
                U[i][j] -= m * U[k][j]                        # zero out below
    return L, U


def py_cholesky(A):
    """Cholesky of an SPD matrix. Returns lower L with L @ L.T == A."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):                                # up to diagonal
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = (A[i][i] - s) ** 0.5                # diagonal: sqrt
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]             # below diagonal
    return L


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_qr(A):
    """QR with NumPy. Returns (Q, R)."""
    return np.linalg.qr(A)


def np_cholesky(A):
    """Cholesky with NumPy (lower L). A must be symmetric positive-definite."""
    return np.linalg.cholesky(A)


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_qr(A):
    """QR with PyTorch. Returns (Q, R)."""
    import torch
    return torch.linalg.qr(A)


def torch_cholesky(A):
    """Cholesky with PyTorch (lower L)."""
    import torch
    return torch.linalg.cholesky(A)


# ======================================================================
# TESTS — do not edit
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


# --- pure-python matrix helpers (for Level 1 checks) ---

def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)]
            for i in range(n)]


def _transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def _allclose(A, B, tol=1e-6):
    return all(abs(A[i][j] - B[i][j]) < tol
               for i in range(len(A)) for j in range(len(A[0])))


# --- Level 1: Pure Python ---

def test_py_lu():
    A = [[4.0, 3.0], [6.0, 3.0]]         # no pivoting needed
    L, U = py_lu(A)
    check(_allclose(_matmul(L, U), A), f"py_lu: L@U != A, got L={L}, U={U}")
    check(abs(L[0][1]) < 1e-9, "py_lu: L must be lower triangular")
    check(abs(U[1][0]) < 1e-9, "py_lu: U must be upper triangular")
    check(abs(L[0][0] - 1) < 1e-9 and abs(L[1][1] - 1) < 1e-9,
          "py_lu: L must have 1s on the diagonal")
    print("PASS  py_lu")


def test_py_cholesky():
    A = [[4.0, 2.0], [2.0, 3.0]]         # symmetric positive-definite
    L = py_cholesky(A)
    check(_allclose(_matmul(L, _transpose(L)), A),
          f"py_cholesky: L@L.T != A, got L={L}")
    check(abs(L[0][1]) < 1e-9, "py_cholesky: L must be lower triangular")
    print("PASS  py_cholesky")


# --- Level 2: NumPy ---

def test_np_qr():
    A = np.array([[12.0, -51.0], [6.0, 167.0]])
    Q, R = np_qr(A)
    check(np.allclose(Q @ R, A), "np_qr: Q@R != A")
    check(np.allclose(Q.T @ Q, np.eye(2)), "np_qr: Q not orthogonal")
    check(abs(R[1][0]) < 1e-6, "np_qr: R must be upper triangular")
    print("PASS  np_qr")


def test_np_cholesky():
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    L = np_cholesky(A)
    check(np.allclose(L @ L.T, A), "np_cholesky: L@L.T != A")
    check(abs(L[0][1]) < 1e-9, "np_cholesky: L must be lower triangular")
    print("PASS  np_cholesky")


# --- Level 3: PyTorch ---

def _run_torch_tests():
    import torch
    A = torch.tensor([[12.0, -51.0], [6.0, 167.0]])
    Q, R = torch_qr(A)
    check(torch.allclose(Q @ R, A, atol=1e-5), "torch_qr: Q@R != A")
    check(torch.allclose(Q.T @ Q, torch.eye(2), atol=1e-5),
          "torch_qr: Q not orthogonal")
    check(abs(float(R[1][0])) < 1e-5, "torch_qr: R must be upper triangular")
    print("PASS  torch_qr")

    S = torch.tensor([[4.0, 2.0], [2.0, 3.0]])
    L = torch_cholesky(S)
    check(torch.allclose(L @ L.T, S, atol=1e-5), "torch_cholesky: L@L.T != A")
    check(abs(float(L[0][1])) < 1e-6, "torch_cholesky: L must be lower triangular")
    print("PASS  torch_cholesky")


def run_torch_tests():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP  PyTorch tests — torch not installed (pip install torch)")
        return
    _run_torch_tests()


if __name__ == "__main__":
    print("\n── Level 1: Pure Python ──")
    test_py_lu()
    test_py_cholesky()

    print("\n── Level 2: NumPy ──")
    test_np_qr()
    test_np_cholesky()

    print("\n── Level 3: PyTorch ──")
    run_torch_tests()

    print("\nAll tests passed!")
