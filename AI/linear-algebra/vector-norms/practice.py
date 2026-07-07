"""
============================================================
  AI / Linear Algebra — Vector & Matrix Norms
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
  LEVEL 2 — NumPy         (np.linalg.norm)
  LEVEL 3 — PyTorch       (torch.linalg.norm)

Each level does the same 5 things:
  1. L1 norm          → sum of absolute values
  2. L2 norm          → sqrt of sum of squares  (the default "length")
  3. L-infinity norm  → largest absolute value
  4. Frobenius norm   → L2 over all entries of a matrix
  5. normalize        → divide a vector by its L2 norm (unit vector)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, using loops)
# ======================================================================
# Vectors are plain lists. Matrices are lists of lists. No numpy/torch.

def py_l1(v):
    """
    L1 norm: sum of absolute values.

    HINT: sum(abs(x) for x in v)
    """
    # TODO
    pass


def py_l2(v):
    """
    L2 norm: square root of the sum of squares.

    HINT: sum(x * x for x in v) ** 0.5     (** 0.5 is square root)
    """
    # TODO
    pass


def py_linf(v):
    """
    L-infinity norm: the largest absolute value.

    HINT: max(abs(x) for x in v)
    """
    # TODO
    pass


def py_frobenius(M):
    """
    Frobenius norm: L2 over every entry of the matrix M (list of rows).

    HINT: sum(x * x for row in M for x in row) ** 0.5
    """
    # TODO
    pass


def py_normalize(v):
    """
    Return v divided by its L2 norm → a unit vector (length 1).

    HINT:
      n = py_l2(v)
      return [x / n for x in v]
      (assume n != 0 for this exercise)
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_l1(v):
    """
    L1 norm with NumPy.

    HINT: np.linalg.norm(v, ord=1)
    """
    # TODO
    pass


def np_l2(v):
    """
    L2 norm with NumPy.

    HINT: np.linalg.norm(v)   (L2 is the default)
    """
    # TODO
    pass


def np_linf(v):
    """
    L-infinity norm with NumPy.

    HINT: np.linalg.norm(v, ord=np.inf)
    """
    # TODO
    pass


def np_frobenius(M):
    """
    Frobenius norm of a matrix with NumPy.

    HINT: np.linalg.norm(M, ord='fro')  (or just np.linalg.norm(M) for 2-D)
    """
    # TODO
    pass


def np_normalize(v):
    """
    Return v divided by its L2 norm.

    HINT: v / np.linalg.norm(v)
    """
    # TODO
    pass


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_l1(v):
    """
    L1 norm with PyTorch.

    HINT: torch.linalg.norm(v, ord=1)   (v must be a float tensor)
    """
    # TODO
    pass


def torch_l2(v):
    """
    L2 norm with PyTorch.

    HINT: torch.linalg.norm(v)   (L2 default)
    """
    # TODO
    pass


def torch_linf(v):
    """
    L-infinity norm with PyTorch.

    HINT: torch.linalg.norm(v, ord=float('inf'))
    """
    # TODO
    pass


def torch_frobenius(M):
    """
    Frobenius norm of a matrix with PyTorch.

    HINT: torch.linalg.norm(M, ord='fro')  (or torch.linalg.norm(M) for 2-D)
    """
    # TODO
    pass


def torch_normalize(v):
    """
    Return v divided by its L2 norm.

    HINT: v / torch.linalg.norm(v)
      (torch.nn.functional.normalize(v, dim=0) also works)
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


# v = [3, -4]:  L1=7, L2=5, Linf=4, normalized=[0.6, -0.8]
# M = [[1,2],[3,4]]:  Frobenius = sqrt(30) ≈ 5.4772

# --- Level 1: Pure Python ---

def test_py():
    v = [3, -4]
    M = [[1, 2], [3, 4]]
    check(close(py_l1(v), 7), f"py_l1 wrong: {py_l1(v)}, want 7")
    print("PASS  py_l1")
    check(close(py_l2(v), 5), f"py_l2 wrong: {py_l2(v)}, want 5")
    print("PASS  py_l2")
    check(close(py_linf(v), 4), f"py_linf wrong: {py_linf(v)}, want 4")
    print("PASS  py_linf")
    check(close(py_frobenius(M), 30 ** 0.5), f"py_frobenius wrong: {py_frobenius(M)}")
    print("PASS  py_frobenius")
    out = py_normalize(v)
    check(close(out[0], 0.6) and close(out[1], -0.8), f"py_normalize wrong: {out}")
    print("PASS  py_normalize")


# --- Level 2: NumPy ---

def test_np():
    v = np.array([3.0, -4.0])
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    check(close(float(np_l1(v)), 7), f"np_l1 wrong: {np_l1(v)}")
    print("PASS  np_l1")
    check(close(float(np_l2(v)), 5), f"np_l2 wrong: {np_l2(v)}")
    print("PASS  np_l2")
    check(close(float(np_linf(v)), 4), f"np_linf wrong: {np_linf(v)}")
    print("PASS  np_linf")
    check(close(float(np_frobenius(M)), 30 ** 0.5), f"np_frobenius wrong: {np_frobenius(M)}")
    print("PASS  np_frobenius")
    out = np_normalize(v)
    check(np.allclose(out, [0.6, -0.8]), f"np_normalize wrong: {out}")
    print("PASS  np_normalize")


# --- Level 3: PyTorch ---

def _run_torch_tests():
    import torch
    v = torch.tensor([3.0, -4.0])
    M = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    check(close(float(torch_l1(v)), 7), f"torch_l1 wrong: {torch_l1(v)}")
    print("PASS  torch_l1")
    check(close(float(torch_l2(v)), 5), f"torch_l2 wrong: {torch_l2(v)}")
    print("PASS  torch_l2")
    check(close(float(torch_linf(v)), 4), f"torch_linf wrong: {torch_linf(v)}")
    print("PASS  torch_linf")
    check(close(float(torch_frobenius(M)), 30 ** 0.5), f"torch_frobenius wrong: {torch_frobenius(M)}")
    print("PASS  torch_frobenius")
    out = torch_normalize(v)
    check(torch.allclose(out, torch.tensor([0.6, -0.8])), f"torch_normalize wrong: {out}")
    print("PASS  torch_normalize")


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
