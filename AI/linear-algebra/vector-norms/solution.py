"""
============================================================
  AI / Linear Algebra — Vector & Matrix Norms
  SOLUTION FILE  (reference)
  Drop-in replacement for practice.py — same signatures.
  Read this only after you have tried practice.py yourself.
============================================================

Requirements:  pip install numpy torch

Levels:
  LEVEL 1 — Pure Python   (loops, no libraries)
  LEVEL 2 — NumPy         (np.linalg.norm)
  LEVEL 3 — PyTorch       (torch.linalg.norm)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, using loops)
# ======================================================================

def py_l1(v):
    """L1 norm: sum of absolute values."""
    return sum(abs(x) for x in v)


def py_l2(v):
    """L2 norm: sqrt of the sum of squares. ** 0.5 is the square root."""
    return sum(x * x for x in v) ** 0.5


def py_linf(v):
    """L-infinity norm: the largest absolute value."""
    return max(abs(x) for x in v)


def py_frobenius(M):
    """Frobenius norm: L2 over every entry of the matrix (double loop)."""
    return sum(x * x for row in M for x in row) ** 0.5


def py_normalize(v):
    """Divide v by its L2 norm → unit vector. Assumes norm != 0."""
    n = py_l2(v)
    return [x / n for x in v]


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_l1(v):
    """L1 norm with NumPy."""
    return np.linalg.norm(v, ord=1)


def np_l2(v):
    """L2 norm with NumPy (the default)."""
    return np.linalg.norm(v)


def np_linf(v):
    """L-infinity norm with NumPy."""
    return np.linalg.norm(v, ord=np.inf)


def np_frobenius(M):
    """Frobenius norm of a matrix. 'fro' is the default for a 2-D array."""
    return np.linalg.norm(M, ord='fro')


def np_normalize(v):
    """Divide v by its L2 norm."""
    return v / np.linalg.norm(v)


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_l1(v):
    """L1 norm with PyTorch. v must be a float tensor."""
    import torch
    return torch.linalg.norm(v, ord=1)


def torch_l2(v):
    """L2 norm with PyTorch (the default)."""
    import torch
    return torch.linalg.norm(v)


def torch_linf(v):
    """L-infinity norm with PyTorch."""
    import torch
    return torch.linalg.norm(v, ord=float('inf'))


def torch_frobenius(M):
    """Frobenius norm of a matrix with PyTorch."""
    import torch
    return torch.linalg.norm(M, ord='fro')


def torch_normalize(v):
    """Divide v by its L2 norm."""
    import torch
    return v / torch.linalg.norm(v)


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

    print("\nAll tests passed!")
