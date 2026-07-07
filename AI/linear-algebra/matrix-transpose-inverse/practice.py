"""
============================================================
  AI / Linear Algebra — Matrix Transpose, Inverse & Multiply
  PRACTICE FILE
  Write every function yourself. Tests tell you if you got
  it right. Do NOT open solution.py first.
============================================================

HOW TO USE
----------
1. Read the hint for each function.
2. Delete the `pass` and write your code.
3. Run:  python practice.py
4. A test PASS means your logic is correct. Fix until all pass.
5. Only open solution.py after you finish or are truly stuck.

Requirements:  pip install numpy torch
   (If torch is missing, the Level 3 tests print SKIP instead
    of failing — but install it to really practice PyTorch.)

This file covers three levels:
  LEVEL 1 — Pure Python   (loops, no libraries — learn the math)
  LEVEL 2 — NumPy         (A.T, np.linalg.det, np.linalg.inv)
  LEVEL 3 — PyTorch       (A.T, torch.linalg.det, torch.linalg.inv)

Each level does the same 5 things:
  1. transpose        → flip rows and columns
  2. matmul           → matrix multiply
  3. determinant      → one number; 0 means no inverse (2x2 only for pure Python)
  4. inverse          → the "undo" of a matrix        (2x2 only for pure Python)
  5. is_identity      → check A @ A_inv ≈ identity
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, using loops)
# ======================================================================
# Matrices are lists of rows. det/inv here handle 2x2 only.

def py_transpose(A):
    """
    Flip rows and columns. Entry [j][i] comes from [i][j].

    HINT:
      rows, cols = len(A), len(A[0])
      return [[A[i][j] for i in range(rows)] for j in range(cols)]
    """
    # TODO
    pass


def py_matmul(A, B):
    """
    Matrix multiply. A is (m,n), B is (n,p). Return (m,p).

    HINT:
      out[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
      Loop over rows of A and columns of B.
    """
    # TODO
    pass


def py_det2(A):
    """
    Determinant of a 2x2 matrix [[a,b],[c,d]] → a*d - b*c.

    HINT:
      (a, b), (c, d) = A
      return a * d - b * c
    """
    # TODO
    pass


def py_inv2(A):
    """
    Inverse of a 2x2 matrix. Raise ValueError if determinant is 0.

    HINT:
      (a, b), (c, d) = A
      det = a*d - b*c
      if det == 0: raise ValueError("singular")
      swap diagonal, negate off-diagonal, divide by det:
        [[ d/det, -b/det],
         [-c/det,  a/det]]
    """
    # TODO
    pass


def py_is_identity(M, tol=1e-6):
    """
    Return True if square matrix M is (close to) the identity.

    HINT:
      For each i, j: expected = 1 if i == j else 0.
      Return False if any abs(M[i][j] - expected) > tol. Else True.
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_transpose(A):
    """
    Transpose with NumPy.

    HINT: A.T
    """
    # TODO
    pass


def np_matmul(A, B):
    """
    Matrix multiply with NumPy.

    HINT: A @ B
    """
    # TODO
    pass


def np_det(A):
    """
    Determinant with NumPy (works for any square size).

    HINT: np.linalg.det(A)
    """
    # TODO
    pass


def np_inv(A):
    """
    Inverse with NumPy.

    HINT: np.linalg.inv(A)
    """
    # TODO
    pass


def np_is_identity(M, tol=1e-6):
    """
    Return True if M is close to the identity matrix.

    HINT: np.allclose(M, np.eye(M.shape[0]), atol=tol)
    """
    # TODO
    pass


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_transpose(A):
    """
    Transpose with PyTorch.

    HINT: A.T
    """
    # TODO
    pass


def torch_matmul(A, B):
    """
    Matrix multiply with PyTorch.

    HINT: A @ B  (or torch.matmul(A, B))
    """
    # TODO
    pass


def torch_det(A):
    """
    Determinant with PyTorch (float tensor required).

    HINT: torch.linalg.det(A)
    """
    # TODO
    pass


def torch_inv(A):
    """
    Inverse with PyTorch.

    HINT: torch.linalg.inv(A)
    """
    # TODO
    pass


def torch_is_identity(M, tol=1e-6):
    """
    Return True if M is close to the identity matrix.

    HINT:
      import torch
      return torch.allclose(M, torch.eye(M.shape[0]), atol=tol)
    """
    # TODO
    pass


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

    print("\nDone.")
