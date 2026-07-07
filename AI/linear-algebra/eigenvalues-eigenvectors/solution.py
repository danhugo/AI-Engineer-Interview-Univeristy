"""
============================================================
  AI / Linear Algebra — Eigenvalues & Eigenvectors
  SOLUTION FILE  (reference)
  Drop-in replacement for practice.py — same signatures.
  Read this only after you have tried practice.py yourself.
============================================================

Requirements:  pip install numpy torch

Levels:
  LEVEL 1 — Pure Python   (power iteration + 2x2 formula)
  LEVEL 2 — NumPy         (np.linalg.eig)
  LEVEL 3 — PyTorch       (torch.linalg.eig)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch)
# ======================================================================

def dot(a, b):
    """Dot product helper."""
    return sum(x * y for x, y in zip(a, b))


def matvec(A, v):
    """Matrix times vector helper."""
    return [dot(row, v) for row in A]


def l2(v):
    """L2 norm helper."""
    return sum(x * x for x in v) ** 0.5


def power_iteration(A, iters=100):
    """
    Find the dominant eigenvalue and its unit eigenvector.
    Each multiply amplifies the largest-eigenvalue direction the most,
    so v converges to that eigenvector.
    """
    v = [1.0] * len(A)                       # any non-zero start works
    for _ in range(iters):
        Av = matvec(A, v)
        n = l2(Av)
        v = [x / n for x in Av]              # normalize → keep unit length
    lam = dot(v, matvec(A, v))               # Rayleigh quotient (v is unit)
    return lam, v


def eig_2x2(A):
    """
    Two eigenvalues of a 2x2 matrix from  λ² − trace·λ + det = 0.
    """
    a, b = A[0]
    c, d = A[1]
    trace = a + d                            # sum of the diagonal
    det = a * d - b * c                      # determinant
    disc = (trace * trace - 4 * det) ** 0.5  # sqrt of discriminant
    return [(trace + disc) / 2, (trace - disc) / 2]


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_eig(A):
    """Eigenvalues and eigenvectors. vecs[:, i] goes with vals[i]."""
    return np.linalg.eig(A)


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_eig(A):
    """Eigenvalues and eigenvectors (complex tensors)."""
    import torch
    return torch.linalg.eig(A)


# ======================================================================
# TESTS — do not edit
# ======================================================================
# Eigenvectors have sign/scale ambiguity and eigenvalues come in any
# order. So we NEVER compare vectors directly. We check either:
#   - the defining property  A v ≈ λ v
#   - sorted eigenvalue lists with a tolerance

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-4):
    return abs(a - b) < tol


# --- Level 1: Pure Python ---

def test_power_iteration():
    # A = [[2,1],[1,2]]: dominant eigenvalue is 3, eigenvector ∝ [1,1]
    A = [[2.0, 1.0], [1.0, 2.0]]
    lam, v = power_iteration(A, iters=200)
    check(close(lam, 3.0), f"dominant eigenvalue wrong: {lam}, want 3")
    # verify defining property: A v ≈ λ v
    Av = matvec(A, v)
    check(all(close(Av[i], lam * v[i]) for i in range(len(v))),
          f"A v != λ v : Av={Av}, λv={[lam*x for x in v]}")
    check(close(l2(v), 1.0), f"eigenvector not unit length: {l2(v)}")
    print("PASS  power_iteration")


def test_eig_2x2():
    # A = [[2,1],[1,2]]: eigenvalues are 3 and 1
    A = [[2.0, 1.0], [1.0, 2.0]]
    vals = sorted(eig_2x2(A))
    check(close(vals[0], 1.0) and close(vals[1], 3.0),
          f"eig_2x2 wrong: {vals}, want [1, 3]")
    print("PASS  eig_2x2")


# --- Level 2: NumPy ---

def test_np_eig():
    A = np.array([[2.0, 0.0], [0.0, 3.0]])   # eigenvalues 2 and 3
    vals, vecs = np_eig(A)
    got = sorted(float(x.real) for x in vals)
    check(close(got[0], 2.0) and close(got[1], 3.0),
          f"np eigenvalues wrong: {got}, want [2, 3]")
    # verify defining property for each pair: A v ≈ λ v
    for i in range(len(vals)):
        v = vecs[:, i]
        check(np.allclose(A @ v, vals[i] * v),
              f"np: A v != λ v for pair {i}")
    print("PASS  np_eig")


# --- Level 3: PyTorch ---

def _run_torch_tests():
    import torch
    A = torch.tensor([[2.0, 0.0], [0.0, 3.0]])   # eigenvalues 2 and 3
    vals, vecs = torch_eig(A)
    got = sorted(float(x.real) for x in vals)
    check(close(got[0], 2.0) and close(got[1], 3.0),
          f"torch eigenvalues wrong: {got}, want [2, 3]")
    # verify defining property: A v ≈ λ v  (use complex A to match dtypes)
    Ac = A.to(vecs.dtype)
    for i in range(len(vals)):
        v = vecs[:, i]
        check(torch.allclose(Ac @ v, vals[i] * v, atol=1e-4),
              f"torch: A v != λ v for pair {i}")
    print("PASS  torch_eig")


def run_torch_tests():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP  PyTorch tests — torch not installed (pip install torch)")
        return
    _run_torch_tests()


if __name__ == "__main__":
    print("\n── Level 1: Pure Python ──")
    test_power_iteration()
    test_eig_2x2()

    print("\n── Level 2: NumPy ──")
    test_np_eig()

    print("\n── Level 3: PyTorch ──")
    run_torch_tests()

    print("\nAll tests passed!")
