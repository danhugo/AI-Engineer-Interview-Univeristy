"""
============================================================
  AI / Linear Algebra — Orthogonality & Projections
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
  LEVEL 2 — NumPy         (@ for dot, np.linalg.norm)
  LEVEL 3 — PyTorch       (torch.dot, torch.linalg.norm)

Each level does the same 3 things:
  1. is_orthogonal(a, b)   → True if the dot product is ~0 (perpendicular)
  2. project(a, b)         → projection (shadow) of a onto b, a vector
  3. gram_schmidt(vectors) → turn vectors into an orthonormal set
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, using loops)
# ======================================================================
# Vectors are plain lists. No numpy / torch.

def py_is_orthogonal(a, b, tol=1e-9):
    """
    Return True if a and b are perpendicular (dot product ~ 0).

    HINT:
      dot = sum(x * y for x, y in zip(a, b))
      return abs(dot) < tol
    """
    # TODO
    pass


def py_project(a, b):
    """
    Projection of a onto b:  (a·b / b·b) * b.  Return a list.

    HINT:
      ab = sum(x * y for x, y in zip(a, b))
      bb = sum(y * y for y in b)
      scale = ab / bb
      return [scale * y for y in b]
    """
    # TODO
    pass


def py_gram_schmidt(vectors):
    """
    Turn a list of vectors into an orthonormal set (list of lists).

    HINT:
      result = []
      for v in vectors:
          w = list(v)                    # copy
          for u in result:               # subtract projection onto each done vector
              p = py_project(w, u)
              w = [wi - pi for wi, pi in zip(w, p)]
          # normalize w (divide by its L2 length) and append
          n = sum(x * x for x in w) ** 0.5
          result.append([x / n for x in w])
      return result
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_is_orthogonal(a, b):
    """
    Return True if a and b are perpendicular.

    HINT: return bool(np.isclose(a @ b, 0.0))
    """
    # TODO
    pass


def np_project(a, b):
    """
    Projection of a onto b as a NumPy array.

    HINT: return (a @ b) / (b @ b) * b
    """
    # TODO
    pass


def np_gram_schmidt(vectors):
    """
    Orthonormal set from a list of 1-D arrays. Return a list of arrays.

    HINT:
      out = []
      for v in vectors:
          w = v.astype(float).copy()
          for u in out:
              w = w - (w @ u) / (u @ u) * u
          out.append(w / np.linalg.norm(w))
      return out
    """
    # TODO
    pass


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_is_orthogonal(a, b):
    """
    Return True if a and b are perpendicular.

    HINT: return bool(torch.isclose(torch.dot(a, b), torch.tensor(0.0)))
    """
    # TODO
    pass


def torch_project(a, b):
    """
    Projection of a onto b as a tensor.

    HINT: return torch.dot(a, b) / torch.dot(b, b) * b
    """
    # TODO
    pass


def torch_gram_schmidt(vectors):
    """
    Orthonormal set from a list of 1-D tensors. Return a list of tensors.

    HINT:
      out = []
      for v in vectors:
          w = v.clone().float()
          for u in out:
              w = w - torch.dot(w, u) / torch.dot(u, u) * u
          out.append(w / torch.linalg.norm(w))
      return out
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


# --- Level 1: Pure Python ---

def test_py():
    check(py_is_orthogonal([1, 0], [0, 1]) is True, "py_is_orthogonal([1,0],[0,1]) should be True")
    check(py_is_orthogonal([1, 1], [1, 0]) is False, "py_is_orthogonal([1,1],[1,0]) should be False")
    print("PASS  py_is_orthogonal")

    out = py_project([3, 3], [1, 0])
    check(close(out[0], 3) and close(out[1], 0), f"py_project([3,3],[1,0]) wrong: {out}")
    out = py_project([2, 4], [1, 0])
    check(close(out[0], 2) and close(out[1], 0), f"py_project([2,4],[1,0]) wrong: {out}")
    print("PASS  py_project")

    basis = py_gram_schmidt([[1, 1], [1, 0]])
    # each vector must be unit length, and the two must be perpendicular
    for u in basis:
        check(close(sum(x * x for x in u) ** 0.5, 1.0), f"py_gram_schmidt not unit length: {u}")
    dot01 = sum(x * y for x, y in zip(basis[0], basis[1]))
    check(close(dot01, 0.0), f"py_gram_schmidt vectors not orthogonal: dot = {dot01}")
    print("PASS  py_gram_schmidt")


# --- Level 2: NumPy ---

def test_np():
    check(np_is_orthogonal(np.array([1., 0.]), np.array([0., 1.])) is True, "np_is_orthogonal True case")
    check(np_is_orthogonal(np.array([1., 1.]), np.array([1., 0.])) is False, "np_is_orthogonal False case")
    print("PASS  np_is_orthogonal")

    check(np.allclose(np_project(np.array([3., 3.]), np.array([1., 0.])), [3., 0.]), "np_project [3,3] onto [1,0]")
    check(np.allclose(np_project(np.array([2., 4.]), np.array([1., 0.])), [2., 0.]), "np_project [2,4] onto [1,0]")
    print("PASS  np_project")

    basis = np_gram_schmidt([np.array([1., 1.]), np.array([1., 0.])])
    for u in basis:
        check(close(float(np.linalg.norm(u)), 1.0), f"np_gram_schmidt not unit length: {u}")
    check(close(float(basis[0] @ basis[1]), 0.0), f"np_gram_schmidt not orthogonal: {basis[0] @ basis[1]}")
    print("PASS  np_gram_schmidt")


# --- Level 3: PyTorch ---

def _run_torch_tests():
    import torch
    check(torch_is_orthogonal(torch.tensor([1., 0.]), torch.tensor([0., 1.])) is True, "torch_is_orthogonal True case")
    check(torch_is_orthogonal(torch.tensor([1., 1.]), torch.tensor([1., 0.])) is False, "torch_is_orthogonal False case")
    print("PASS  torch_is_orthogonal")

    check(torch.allclose(torch_project(torch.tensor([3., 3.]), torch.tensor([1., 0.])), torch.tensor([3., 0.])),
          "torch_project [3,3] onto [1,0]")
    check(torch.allclose(torch_project(torch.tensor([2., 4.]), torch.tensor([1., 0.])), torch.tensor([2., 0.])),
          "torch_project [2,4] onto [1,0]")
    print("PASS  torch_project")

    basis = torch_gram_schmidt([torch.tensor([1., 1.]), torch.tensor([1., 0.])])
    for u in basis:
        check(close(float(torch.linalg.norm(u)), 1.0), f"torch_gram_schmidt not unit length: {u}")
    check(close(float(torch.dot(basis[0], basis[1])), 0.0),
          f"torch_gram_schmidt not orthogonal: {torch.dot(basis[0], basis[1])}")
    print("PASS  torch_gram_schmidt")


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
