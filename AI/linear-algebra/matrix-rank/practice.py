"""
============================================================
  AI / Linear Algebra — Rank of a Matrix
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
  LEVEL 1 — Pure Python   (Gaussian elimination — learn the math)
  LEVEL 2 — NumPy         (np.linalg.matrix_rank)
  LEVEL 3 — PyTorch       (torch.linalg.matrix_rank)

Rank = number of linearly independent rows = how many rows carry
real, non-redundant information.
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, Gaussian elimination)
# ======================================================================
# M is a list of rows (list of lists). Return an int rank.

def rank_by_elimination(M, tol=1e-9):
    """
    Rank by row reduction: count how many pivots you can place.

    HINT:
      1. Copy M into floats so you don't change the input:
           A = [[float(x) for x in row] for row in M]
      2. Track `row` = next row to hold a pivot (start 0), `rank` = 0.
      3. Loop `col` over every column:
           - if row >= number_of_rows: break
           - find a row i at/below `row` with abs(A[i][col]) > tol
           - if none found: continue (no pivot in this column)
           - swap that row up to position `row`
           - for each row below, subtract a multiple to zero out this column:
               factor = A[i][col] / A[row][col]
               A[i][j] -= factor * A[row][j]   for j from col onward
           - rank += 1 ; row += 1
      4. return rank
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_rank(M):
    """
    Rank with NumPy. Return an int.

    HINT: int(np.linalg.matrix_rank(M))
    """
    # TODO
    pass


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_rank(M):
    """
    Rank with PyTorch. Return an int. (M must be a float tensor.)

    HINT: int(torch.linalg.matrix_rank(M))
    """
    # TODO
    pass


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

    print("\nDone.")
