"""
============================================================
  AI / Linear Algebra — Matrix Transpose, Inverse & Multiply
  SOLUTION FILE  (reference)
  Drop-in replacement for practice.py — same signatures.
  Read this only after you have tried practice.py yourself.
============================================================

Requirements:  pip install numpy torch

Levels:
  LEVEL 1 — Pure Python   (loops, no libraries; 2x2 det/inv)
  LEVEL 2 — NumPy         (A.T, np.linalg.det, np.linalg.inv)
  LEVEL 3 — PyTorch       (A.T, torch.linalg.det, torch.linalg.inv)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, using loops)
# ======================================================================

def py_transpose(A):
    """Flip rows and columns: entry [j][i] comes from [i][j]."""
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def py_matmul(A, B):
    """Matrix multiply (m,n)·(n,p) → (m,p). out[i][j] = row i · col j."""
    n, p = len(B), len(B[0])
    out = []
    for row in A:
        out.append([sum(row[k] * B[k][j] for k in range(n)) for j in range(p)])
    return out


def py_det2(A):
    """Determinant of a 2x2: a*d - b*c."""
    (a, b), (c, d) = A
    return a * d - b * c


def py_inv2(A):
    """Inverse of a 2x2. Swap diagonal, negate off-diagonal, divide by det."""
    (a, b), (c, d) = A
    det = a * d - b * c
    if det == 0:
        raise ValueError("singular matrix — no inverse")
    return [[d / det, -b / det],
            [-c / det, a / det]]


def py_is_identity(M, tol=1e-6):
    """True if square M is close to the identity (1s on diagonal, else 0)."""
    n = len(M)
    for i in range(n):
        for j in range(n):
            expected = 1 if i == j else 0
            if abs(M[i][j] - expected) > tol:
                return False
    return True


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_transpose(A):
    """Transpose with NumPy."""
    return A.T


def np_matmul(A, B):
    """Matrix multiply with NumPy."""
    return A @ B


def np_det(A):
    """Determinant with NumPy (any square size)."""
    return np.linalg.det(A)


def np_inv(A):
    """Inverse with NumPy."""
    return np.linalg.inv(A)


def np_is_identity(M, tol=1e-6):
    """True if M is close to the identity matrix."""
    return np.allclose(M, np.eye(M.shape[0]), atol=tol)


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_transpose(A):
    """Transpose with PyTorch."""
    return A.T


def torch_matmul(A, B):
    """Matrix multiply with PyTorch."""
    return A @ B


def torch_det(A):
    """Determinant with PyTorch (float tensor required)."""
    import torch
    return torch.linalg.det(A)


def torch_inv(A):
    """Inverse with PyTorch."""
    import torch
    return torch.linalg.inv(A)


def torch_is_identity(M, tol=1e-6):
    """True if M is close to the identity matrix."""
    import torch
    return torch.allclose(M, torch.eye(M.shape[0]), atol=tol)


# ======================================================================
# TESTS — do not edit
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


# A = [[4,7],[2,6]]:  det = 10
# A_inv = [[0.6,-0.7],[-0.2,0.4]]
# A @ A_inv = identity

# --- Level 1: Pure Python ---

def test_py():
    A = [[4, 7], [2, 6]]
    B = [[1, 2], [3, 4]]

    check(py_transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]],
          f"py_transpose wrong: {py_transpose([[1, 2, 3], [4, 5, 6]])}")
    print("PASS  py_transpose")

    check(py_matmul(A, B) == [[25, 36], [20, 28]], f"py_matmul wrong: {py_matmul(A, B)}")
    print("PASS  py_matmul")

    check(close(py_det2(A), 10), f"py_det2 wrong: {py_det2(A)}, want 10")
    print("PASS  py_det2")

    inv = py_inv2(A)
    check(close(inv[0][0], 0.6) and close(inv[0][1], -0.7)
          and close(inv[1][0], -0.2) and close(inv[1][1], 0.4),
          f"py_inv2 wrong: {inv}")
    print("PASS  py_inv2")

    raised = False
    try:
        py_inv2([[1, 2], [2, 4]])   # det = 0
    except ValueError:
        raised = True
    check(raised, "py_inv2 should raise ValueError on a singular matrix")
    print("PASS  py_inv2 singular raises")

    prod = py_matmul(A, py_inv2(A))
    check(py_is_identity(prod), f"py_is_identity: A @ A_inv should be identity, got {prod}")
    check(py_is_identity([[1, 0], [0, 1]]) and not py_is_identity([[1, 1], [0, 1]]),
          "py_is_identity basic checks failed")
    print("PASS  py_is_identity")


# --- Level 2: NumPy ---

def test_np():
    A = np.array([[4., 7.], [2., 6.]])

    check(np.array_equal(np_transpose(np.array([[1, 2, 3], [4, 5, 6]])),
                         np.array([[1, 4], [2, 5], [3, 6]])),
          "np_transpose wrong")
    print("PASS  np_transpose")

    B = np.array([[1., 2.], [3., 4.]])
    check(np.allclose(np_matmul(A, B), np.array([[25., 36.], [20., 28.]])),
          f"np_matmul wrong: {np_matmul(A, B)}")
    print("PASS  np_matmul")

    check(close(float(np_det(A)), 10), f"np_det wrong: {np_det(A)}")
    print("PASS  np_det")

    inv = np_inv(A)
    check(np.allclose(inv, np.array([[0.6, -0.7], [-0.2, 0.4]])), f"np_inv wrong: {inv}")
    print("PASS  np_inv")

    check(np_is_identity(A @ np_inv(A)), "np_is_identity: A @ A_inv should be identity")
    check(np_is_identity(np.eye(2)) and not np_is_identity(np.array([[1., 1.], [0., 1.]])),
          "np_is_identity basic checks failed")
    print("PASS  np_is_identity")


# --- Level 3: PyTorch ---

def _run_torch_tests():
    import torch
    A = torch.tensor([[4., 7.], [2., 6.]])

    check(torch.equal(torch_transpose(torch.tensor([[1, 2, 3], [4, 5, 6]])),
                      torch.tensor([[1, 4], [2, 5], [3, 6]])),
          "torch_transpose wrong")
    print("PASS  torch_transpose")

    B = torch.tensor([[1., 2.], [3., 4.]])
    check(torch.allclose(torch_matmul(A, B), torch.tensor([[25., 36.], [20., 28.]])),
          f"torch_matmul wrong: {torch_matmul(A, B)}")
    print("PASS  torch_matmul")

    check(close(float(torch_det(A)), 10), f"torch_det wrong: {torch_det(A)}")
    print("PASS  torch_det")

    inv = torch_inv(A)
    check(torch.allclose(inv, torch.tensor([[0.6, -0.7], [-0.2, 0.4]]), atol=1e-5),
          f"torch_inv wrong: {inv}")
    print("PASS  torch_inv")

    check(torch_is_identity(A @ torch_inv(A)), "torch_is_identity: A @ A_inv should be identity")
    check(torch_is_identity(torch.eye(2))
          and not torch_is_identity(torch.tensor([[1., 1.], [0., 1.]])),
          "torch_is_identity basic checks failed")
    print("PASS  torch_is_identity")


def run_torch_tests():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP  PyTorch tests — torch not installed (pip install torch)")
        return
    _run_torch_tests()


if __name__ == "__main__":
    print("\n── Level 1: Pure Python ──")
    test_py()

    print("\n── Level 2: NumPy ──")
    test_np()

    print("\n── Level 3: PyTorch ──")
    run_torch_tests()

    print("\nAll tests passed!")
