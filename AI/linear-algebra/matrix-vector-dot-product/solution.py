"""
============================================================
  AI / Linear Algebra — Matrix, Vector & Dot Product
  SOLUTION FILE  (reference)
  Drop-in replacement for practice.py — same signatures.
  Read this only after you have tried practice.py yourself.
============================================================

Requirements:  pip install numpy torch

Levels:
  LEVEL 1 — Pure Python   (loops, no libraries)
  LEVEL 2 — NumPy         (np.dot, @, matmul)
  LEVEL 3 — PyTorch       (torch.dot, torch.mv, torch.matmul)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, using loops)
# ======================================================================

def py_dot(a, b):
    """Dot product of two lists → one number."""
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]      # multiply matching elements, add up
    return total


def py_matvec(W, x):
    """Matrix × vector: one dot product per row of W."""
    return [py_dot(row, x) for row in W]


def py_matmul(A, B):
    """
    Matrix × matrix from scratch.
      out[i][j] = row i of A  dotted with  column j of B
    """
    n = len(B)            # shared inner dimension
    p = len(B[0])         # number of columns in B
    out = []
    for row in A:
        new_row = []
        for j in range(p):
            col = [B[k][j] for k in range(n)]   # column j of B
            new_row.append(py_dot(row, col))
        out.append(new_row)
    return out


def py_linear(X, W, b):
    """
    Batch linear layer, from scratch:  Y = X @ W.T + b
    Dotting each input row with each weight row already does the transpose.
    """
    out = []
    for x in X:
        row = [py_dot(x, W[j]) + b[j] for j in range(len(W))]
        out.append(row)
    return out


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_dot(a, b):
    """Dot product of two 1-D arrays → one number."""
    return a @ b            # @ multiplies element-wise then sums


def np_matvec(W, x):
    """Matrix (m,n) times vector (n,) → (m,). One row·x per output entry."""
    return W @ x


def np_matmul(A, B):
    """Matrix (m,n) times matrix (n,p) → (m,p). Use @, not * (elementwise)."""
    return A @ B


def np_linear(X, W, b):
    """
    Batch linear layer:  Y = X @ W.T + b
      X (batch, in), W (out, in), b (out,)  →  Y (batch, out)
    W.T turns (out, in) into (in, out) so the inner dims match.
    b broadcasts onto every row.
    """
    return X @ W.T + b


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_dot(a, b):
    """Dot product of two 1-D tensors → scalar tensor. Same dtype required."""
    import torch
    return torch.dot(a, b)


def torch_matvec(W, x):
    """Matrix (m,n) times vector (n,) → (m,). mv = matrix-vector."""
    import torch
    return torch.mv(W, x)


def torch_matmul(A, B):
    """Matrix (m,n) times matrix (n,p) → (m,p)."""
    import torch
    return torch.matmul(A, B)


def torch_linear(X, W, b):
    """Batch linear layer in torch:  X @ W.T + b."""
    return X @ W.T + b


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

    print("\nAll tests passed!")
