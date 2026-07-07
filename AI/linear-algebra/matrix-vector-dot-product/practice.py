"""
============================================================
  AI / Linear Algebra — Matrix, Vector & Dot Product
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
  LEVEL 2 — NumPy         (np.dot, @, matmul)
  LEVEL 3 — PyTorch       (torch.dot, torch.mv, torch.matmul)

Each level does the same 4 things:
  1. vector dot product      → one number
  2. matrix × vector         → one dot product per row
  3. matrix × matrix         → row·column for every pair
  4. linear layer forward    → X @ W.T + b   (batch of inputs)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, using loops)
# ======================================================================
# Inputs are plain Python lists / lists of lists. No numpy, no torch.
# This is where you really understand what the math does.

def py_dot(a, b):
    """
    Dot product of two lists → one number.

    HINT:
      total = 0
      for each index i: total += a[i] * b[i]
      return total
      (a and b have the same length.)
    """
    # TODO
    pass


def py_matvec(W, x):
    """
    Matrix × vector. W is a list of rows (m rows, each length n).
    x is a list of length n. Return a list of length m.

    HINT: for each row in W, compute py_dot(row, x). Collect into a list.
    """
    # TODO
    pass


def py_matmul(A, B):
    """
    Matrix × matrix. A is (m, n), B is (n, p). Return (m, p) as list of lists.

    HINT:
      out[i][j] = row i of A  dotted with  column j of B
      Column j of B is [B[k][j] for k in range(n)].
      Loop i over rows of A, j over columns of B.
    """
    # TODO
    pass


def py_linear(X, W, b):
    """
    Linear layer over a batch, from scratch.
      X is (batch, in),  W is (out, in),  b is (out,)
    Return (batch, out) as a list of lists.

    HINT:
      For each input row x in X, and each weight row w in W:
        out_value = py_dot(x, w) + b[<index of w>]
      This is the same as X @ W.T + b — dotting each input with each
      weight row already does the transpose for you.
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_dot(a, b):
    """
    Dot product of two 1-D arrays. Return a single number.

    HINT: use `a @ b` (or np.dot(a, b)).
    """
    # TODO
    pass


def np_matvec(W, x):
    """
    Matrix times vector. W is (m, n), x is (n,). Return shape (m,).

    HINT: `W @ x`. Each output entry is one row of W dotted with x.
    """
    # TODO
    pass


def np_matmul(A, B):
    """
    Matrix times matrix. A is (m, n), B is (n, p). Return shape (m, p).

    HINT: `A @ B`. Remember: use @, NOT * (which is elementwise).
    """
    # TODO
    pass


def np_linear(X, W, b):
    """
    One linear (dense) layer over a BATCH of inputs.
      X is (batch, in_features)
      W is (out_features, in_features)   <- note the order
      b is (out_features,)
    Return shape (batch, out_features).

    HINT:
      The math is  Y = X @ W.T + b
      - W.T flips W to (in_features, out_features) so shapes line up:
          (batch, in) @ (in, out) = (batch, out)
      - adding b broadcasts it onto every row.
    """
    # TODO
    pass


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================
# These use torch tensors. Same 4 operations as before.

def torch_dot(a, b):
    """
    Dot product of two 1-D tensors. Return a scalar tensor.

    HINT: torch.dot(a, b). Both tensors must be the same dtype (use floats).
    """
    # TODO
    pass


def torch_matvec(W, x):
    """
    Matrix times vector for tensors. W is (m, n), x is (n,). Return (m,).

    HINT: torch.mv(W, x)   (mv = matrix-vector).  `W @ x` also works.
    """
    # TODO
    pass


def torch_matmul(A, B):
    """
    Matrix times matrix for tensors. A is (m, n), B is (n, p). Return (m, p).

    HINT: torch.matmul(A, B)  or  A @ B.
    """
    # TODO
    pass


def torch_linear(X, W, b):
    """
    Linear layer over a batch, in PyTorch.
      X is (batch, in_features)
      W is (out_features, in_features)
      b is (out_features,)
    Return (batch, out_features).

    HINT: same math as np_linear:  X @ W.T + b
      In torch, transpose a 2-D tensor with W.T (or W.transpose(0, 1)).
    """
    # TODO
    pass


# ======================================================================
# TESTS — do not edit
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


# --- Level 1: Pure Python ---

def test_py_dot():
    out = py_dot([1, 2, 3], [4, 5, 6])
    check(out == 32, f"py_dot wrong: got {out}, want 32")
    print("PASS  py_dot")


def test_py_matvec():
    W = [[1, 2, 3],
         [4, 5, 6]]
    x = [1, 0, 1]
    out = py_matvec(W, x)
    check(out == [4, 10], f"py_matvec wrong: {out}")
    print("PASS  py_matvec")


def test_py_matmul():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    out = py_matmul(A, B)
    check(out == [[19, 22], [43, 50]], f"py_matmul wrong: {out}")
    print("PASS  py_matmul")


def test_py_linear():
    X = [[1, 2], [3, 4]]              # (batch=2, in=2)
    W = [[1, 0], [0, 1], [1, 1]]      # (out=3, in=2)
    b = [10, 20, 30]                  # (out=3,)
    out = py_linear(X, W, b)
    check(out == [[11, 22, 33], [13, 24, 37]], f"py_linear wrong: {out}")
    print("PASS  py_linear")


# --- Level 2: NumPy ---

def test_np_dot():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    out = np_dot(a, b)
    check(int(out) == 32, f"np_dot wrong: got {out}, want 32")
    print("PASS  np_dot")


def test_np_matvec():
    W = np.array([[1, 2, 3],
                  [4, 5, 6]])
    x = np.array([1, 0, 1])
    out = np_matvec(W, x)
    check(np.array_equal(out, np.array([4, 10])), f"np_matvec wrong: {out}")
    print("PASS  np_matvec")


def test_np_matmul():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    out = np_matmul(A, B)
    check(np.array_equal(out, np.array([[19, 22], [43, 50]])),
          f"np_matmul wrong: {out}")
    print("PASS  np_matmul")


def test_np_linear():
    X = np.array([[1., 2.],
                  [3., 4.]])          # (batch=2, in=2)
    W = np.array([[1., 0.],
                  [0., 1.],
                  [1., 1.]])          # (out=3, in=2)
    b = np.array([10., 20., 30.])     # (out=3,)
    out = np_linear(X, W, b)
    want = X @ W.T + b
    check(out.shape == (2, 3), f"np_linear shape wrong: {out.shape}, want (2, 3)")
    check(np.allclose(out, want), f"np_linear values wrong: {out}")
    print("PASS  np_linear")


# --- Level 3: PyTorch ---

def _run_torch_tests():
    import torch

    def t_dot():
        a = torch.tensor([1., 2., 3.])
        b = torch.tensor([4., 5., 6.])
        out = torch_dot(a, b)
        check(float(out) == 32.0, f"torch_dot wrong: got {out}, want 32")
        print("PASS  torch_dot")

    def t_matvec():
        W = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.]])
        x = torch.tensor([1., 0., 1.])
        out = torch_matvec(W, x)
        check(torch.allclose(out, torch.tensor([4., 10.])),
              f"torch_matvec wrong: {out}")
        print("PASS  torch_matvec")

    def t_matmul():
        A = torch.tensor([[1., 2.], [3., 4.]])
        B = torch.tensor([[5., 6.], [7., 8.]])
        out = torch_matmul(A, B)
        check(torch.allclose(out, torch.tensor([[19., 22.], [43., 50.]])),
              f"torch_matmul wrong: {out}")
        print("PASS  torch_matmul")

    def t_linear():
        X = torch.tensor([[1., 2.], [3., 4.]])            # (2, 2)
        W = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])  # (3, 2)
        b = torch.tensor([10., 20., 30.])                 # (3,)
        out = torch_linear(X, W, b)
        want = X @ W.T + b
        check(tuple(out.shape) == (2, 3),
              f"torch_linear shape wrong: {tuple(out.shape)}, want (2, 3)")
        check(torch.allclose(out, want), f"torch_linear values wrong: {out}")
        print("PASS  torch_linear")

    t_dot()
    t_matvec()
    t_matmul()
    t_linear()


def run_torch_tests():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP  PyTorch tests — torch not installed (pip install torch)")
        return
    _run_torch_tests()


if __name__ == "__main__":
    print("\n── Level 1: Pure Python ──")
    test_py_dot()
    test_py_matvec()
    test_py_matmul()
    test_py_linear()

    print("\n── Level 2: NumPy ──")
    test_np_dot()
    test_np_matvec()
    test_np_matmul()
    test_np_linear()

    print("\n── Level 3: PyTorch ──")
    run_torch_tests()

    print("\nDone.")
