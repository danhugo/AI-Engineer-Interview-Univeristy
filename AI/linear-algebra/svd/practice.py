"""
============================================================
  AI / Linear Algebra — SVD (Singular Value Decomposition)
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
  LEVEL 2 — NumPy         (np.linalg.svd)
  LEVEL 3 — PyTorch       (torch.linalg.svd)

Idea to remember:  A = U Σ Vᵀ   (rotate → stretch → rotate).
The singular values (diagonal of Σ) are √(eigenvalues of AᵀA).
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, using loops)
# ======================================================================
# Matrices are lists of lists. No numpy / torch here.
# transpose() and matmul() are GIVEN — you built these in earlier topics.
# You only write the SVD-specific parts.

def transpose(A):
    """Flip rows and columns. GIVEN — do not change."""
    return [[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]


def matmul(A, B):
    """Matrix multiply, from scratch. GIVEN — do not change."""
    n, p = len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
            for i in range(len(A))]


def singular_values_via_ata(A):
    """
    Singular values of a 2x2 matrix A, sorted largest first.
    Uses the fact:  σ = √(eigenvalue of AᵀA).

    HINT:
      1. S = matmul(transpose(A), A)   # 2x2 symmetric: [[a, b], [b, c]]
      2. a, b, c = S[0][0], S[0][1], S[1][1]
      3. eigenvalues of a 2x2 symmetric matrix (quadratic formula):
            tr  = a + c
            det = a*c - b*b
            disc = (tr*tr - 4*det) ** 0.5
            l1, l2 = (tr + disc) / 2, (tr - disc) / 2
      4. singular value = √eigenvalue. Clamp tiny negatives to 0:
            max(l, 0) ** 0.5
      5. return them sorted(..., reverse=True)   # largest first
    """
    # TODO
    pass


def reconstruct(U, s, Vt):
    """
    Rebuild A from its SVD pieces:  A = U @ diag(s) @ Vt.
    U is (m x m), s is a list of singular values, Vt is (n x n).
    Return the (m x n) matrix as a list of lists.

    HINT:
      1. Build Sigma: a matrix same rows as U, same cols as Vt,
         with s[i] on the diagonal (i == j) and 0 elsewhere.
            Sigma = [[s[i] if i == j else 0 for j in range(len(Vt))]
                     for i in range(len(U))]
      2. return matmul(matmul(U, Sigma), Vt)
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_svd(A):
    """
    Return the SVD of A as (U, s, Vt).
      U  → (m, m),  s → 1-D singular values (descending),  Vt → (n, n) = Vᵀ

    HINT: return np.linalg.svd(A)   (it already returns U, s, Vt)
    """
    # TODO
    pass


def np_rank_k_approx(A, k):
    """
    Best rank-k approximation of A: keep only the top-k singular values.
    Return an (m, n) array. With k = full rank you get A back.

    HINT:
      U, s, Vt = np.linalg.svd(A)
      return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    """
    # TODO
    pass


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_svd(A):
    """
    Return the SVD of tensor A as (U, S, Vh).  Vh is Vᵀ, S is 1-D descending.

    HINT: return torch.linalg.svd(A)   (A must be a float tensor)
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


def close_list(xs, ys, tol=1e-6):
    return len(xs) == len(ys) and all(close(x, y, tol) for x, y in zip(xs, ys))


# Test matrices and their known singular values (computed independently):
#   A1 = [[3, 0], [0, 2]]     -> [3, 2]
#   A2 = [[4, 0], [3, -5]]    -> [sqrt(40), sqrt(10)] = [6.32456, 3.16228]
A1 = [[3.0, 0.0], [0.0, 2.0]]
A2 = [[4.0, 0.0], [3.0, -5.0]]
S1 = [3.0, 2.0]
S2 = [40.0 ** 0.5, 10.0 ** 0.5]


# --- Level 1: Pure Python ---

def test_py():
    check(close_list(singular_values_via_ata(A1), S1),
          f"singular_values_via_ata(A1) wrong: {singular_values_via_ata(A1)}")
    print("PASS  singular_values_via_ata (diagonal)")
    check(close_list(singular_values_via_ata(A2), S2, tol=1e-5),
          f"singular_values_via_ata(A2) wrong: {singular_values_via_ata(A2)}")
    print("PASS  singular_values_via_ata (non-diagonal)")

    # reconstruct with identity rotations: U @ diag(s) @ I = diag(s)
    out = reconstruct([[1, 0], [0, 1]], [3, 2], [[1, 0], [0, 1]])
    check(out == [[3, 0], [0, 2]], f"reconstruct (identity) wrong: {out}")
    print("PASS  reconstruct (identity)")

    # reconstruct with a row-swap U: swaps the two rows of diag(s)
    out2 = reconstruct([[0, 1], [1, 0]], [3, 2], [[1, 0], [0, 1]])
    check(out2 == [[0, 2], [3, 0]], f"reconstruct (swap) wrong: {out2}")
    print("PASS  reconstruct (rotation)")


# --- Level 2: NumPy ---

def test_np():
    for A, S in [(np.array(A1), S1), (np.array(A2), S2)]:
        U, s, Vt = np_svd(A)
        check(close_list(sorted(s.tolist(), reverse=True), S, tol=1e-5),
              f"np_svd singular values wrong: {s}")
        rebuilt = U @ np.diag(s) @ Vt
        check(np.allclose(rebuilt, A), f"np_svd reconstruction wrong: {rebuilt}")
    print("PASS  np_svd (singular values + reconstruction)")

    A = np.array(A2)
    full = np_rank_k_approx(A, 2)
    check(full.shape == A.shape, f"rank-k shape wrong: {full.shape}")
    check(np.allclose(full, A), "rank-2 (full) should equal A")
    print("PASS  np_rank_k_approx (full k = A)")

    approx1 = np_rank_k_approx(A, 1)
    check(approx1.shape == A.shape, f"rank-1 shape wrong: {approx1.shape}")
    # rank-1 keeps only the biggest singular value
    s_of_approx = sorted(np.linalg.svd(approx1)[1].tolist(), reverse=True)
    check(close(s_of_approx[0], S2[0], tol=1e-5) and close(s_of_approx[1], 0, tol=1e-5),
          f"rank-1 should keep only top singular value: {s_of_approx}")
    print("PASS  np_rank_k_approx (rank-1 keeps top singular value)")


# --- Level 3: PyTorch ---

def _run_torch_tests():
    import torch
    for A_list, S in [(A1, S1), (A2, S2)]:
        A = torch.tensor(A_list)
        U, s, Vh = torch_svd(A)
        vals = sorted(s.tolist(), reverse=True)
        check(close_list(vals, S, tol=1e-5), f"torch_svd singular values wrong: {s}")
        rebuilt = U @ torch.diag(s) @ Vh
        check(torch.allclose(rebuilt, A, atol=1e-5),
              f"torch_svd reconstruction wrong: {rebuilt}")
    print("PASS  torch_svd (singular values + reconstruction)")


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
