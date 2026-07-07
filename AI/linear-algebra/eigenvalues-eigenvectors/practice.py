"""
============================================================
  AI / Linear Algebra — Eigenvalues & Eigenvectors
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
  LEVEL 1 — Pure Python   (power iteration + 2x2 formula — learn the math)
  LEVEL 2 — NumPy         (np.linalg.eig / eigvals)
  LEVEL 3 — PyTorch       (torch.linalg.eig / eigvals)

Key idea:  A v = λ v
  v = eigenvector (direction the matrix does not turn)
  λ = eigenvalue (how much it stretches)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch)
# ======================================================================
# Matrices are lists of rows. Vectors are plain lists. No numpy/torch.

def dot(a, b):
    """Dot product helper. Provided for you."""
    return sum(x * y for x, y in zip(a, b))


def matvec(A, v):
    """Matrix times vector helper. Provided for you."""
    return [dot(row, v) for row in A]


def l2(v):
    """L2 norm helper. Provided for you."""
    return sum(x * x for x in v) ** 0.5


def power_iteration(A, iters=100):
    """
    Find the DOMINANT eigenvalue and its unit eigenvector.
    Return (lam, v) where v has length 1.

    HINT:
      1. Start with v = [1.0, 1.0, ...] (one 1.0 per row of A).
      2. Repeat `iters` times:
           Av = matvec(A, v)
           n  = l2(Av)
           v  = [x / n for x in Av]        # normalize to unit length
      3. Eigenvalue via Rayleigh quotient (v is unit length):
           lam = dot(v, matvec(A, v))
      4. return lam, v
    """
    # TODO
    pass


def eig_2x2(A):
    """
    Return the two eigenvalues of a 2x2 matrix as a list [λ1, λ2],
    using the characteristic equation  λ² − trace·λ + det = 0.

    HINT:
      a, b = A[0]
      c, d = A[1]
      trace = a + d
      det   = a * d - b * c
      disc  = (trace * trace - 4 * det) ** 0.5
      return [(trace + disc) / 2, (trace - disc) / 2]
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_eig(A):
    """
    Return (vals, vecs) for matrix A using NumPy.
    vals is a 1-D array of eigenvalues.
    vecs[:, i] is the eigenvector for vals[i].

    HINT: return np.linalg.eig(A)
    """
    # TODO
    pass


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_eig(A):
    """
    Return (vals, vecs) for matrix A using PyTorch.
    Results are complex tensors.

    HINT: import torch; return torch.linalg.eig(A)
    """
    # TODO
    pass


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

    print("\nDone.")
