"""
============================================================
  AI / Linear Algebra — Rank of a Matrix
  SOLUTION FILE  (reference)
  Drop-in replacement for practice.py — same signatures.
  Read this only after you have tried practice.py yourself.
============================================================

Requirements:  pip install numpy torch

Levels:
  LEVEL 1 — Pure Python   (Gaussian elimination)
  LEVEL 2 — NumPy         (np.linalg.matrix_rank)
  LEVEL 3 — PyTorch       (torch.linalg.matrix_rank)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, Gaussian elimination)
# ======================================================================

def rank_by_elimination(M, tol=1e-9):
    """
    Rank by row reduction. Each pivot placed = one independent row.
    Redundant rows collapse to zero and never provide a pivot.
    """
    # float copy so we don't change the caller's matrix
    A = [[float(x) for x in row] for row in M]
    rows, cols = len(A), len(A[0])
    rank = 0
    row = 0                          # next row to hold a pivot

    for col in range(cols):
        if row >= rows:
            break
        # find a row at/below `row` with a non-zero entry in this column
        pivot = None
        for i in range(row, rows):
            if abs(A[i][col]) > tol:
                pivot = i
                break
        if pivot is None:
            continue                 # no pivot in this column, move on

        A[row], A[pivot] = A[pivot], A[row]     # swap pivot row up
        # zero out this column in every row below
        for i in range(row + 1, rows):
            factor = A[i][col] / A[row][col]
            for j in range(col, cols):
                A[i][j] -= factor * A[row][j]
        rank += 1
        row += 1

    return rank


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_rank(M):
    """Rank with NumPy. Counts singular values above a tolerance."""
    return int(np.linalg.matrix_rank(M))


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_rank(M):
    """Rank with PyTorch. M must be a float tensor."""
    import torch
    return int(torch.linalg.matrix_rank(M))


# ======================================================================
# TESTS — do not edit
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


# Known ranks:
#   full rank 2:  [[1,2],[3,4]]
#   rank 1:       [[1,2],[2,4]]   (row2 = 2*row1)
#   rank 2 (3x3): [[1,0,0],[0,1,0],[0,0,0]]

FULL   = [[1, 2], [3, 4]]
DEFIC  = [[1, 2], [2, 4]]
THREE  = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]


# --- Level 1: Pure Python ---

def test_py():
    check(rank_by_elimination(FULL) == 2, f"rank FULL wrong: {rank_by_elimination(FULL)}, want 2")
    print("PASS  rank_by_elimination full rank")
    check(rank_by_elimination(DEFIC) == 1, f"rank DEFIC wrong: {rank_by_elimination(DEFIC)}, want 1")
    print("PASS  rank_by_elimination rank-deficient")
    check(rank_by_elimination(THREE) == 2, f"rank THREE wrong: {rank_by_elimination(THREE)}, want 2")
    print("PASS  rank_by_elimination 3x3")


# --- Level 2: NumPy ---

def test_np():
    check(np_rank(np.array(FULL)) == 2, f"np_rank FULL wrong: {np_rank(np.array(FULL))}")
    print("PASS  np_rank full rank")
    check(np_rank(np.array(DEFIC)) == 1, f"np_rank DEFIC wrong: {np_rank(np.array(DEFIC))}")
    print("PASS  np_rank rank-deficient")
    check(np_rank(np.array(THREE)) == 2, f"np_rank THREE wrong: {np_rank(np.array(THREE))}")
    print("PASS  np_rank 3x3")


# --- Level 3: PyTorch ---

def _run_torch_tests():
    import torch
    check(torch_rank(torch.tensor(FULL, dtype=torch.float)) == 2, "torch_rank FULL wrong")
    print("PASS  torch_rank full rank")
    check(torch_rank(torch.tensor(DEFIC, dtype=torch.float)) == 1, "torch_rank DEFIC wrong")
    print("PASS  torch_rank rank-deficient")
    check(torch_rank(torch.tensor(THREE, dtype=torch.float)) == 2, "torch_rank THREE wrong")
    print("PASS  torch_rank 3x3")


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
