"""
============================================================
  AI / Linear Algebra — Orthogonality & Projections
  SOLUTION FILE  (reference)
  Drop-in replacement for practice.py — same signatures.
  Read this only after you have tried practice.py yourself.
============================================================

Requirements:  pip install numpy torch

Levels:
  LEVEL 1 — Pure Python   (loops, no libraries)
  LEVEL 2 — NumPy         (@ for dot, np.linalg.norm)
  LEVEL 3 — PyTorch       (torch.dot, torch.linalg.norm)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — Pure Python (from scratch, using loops)
# ======================================================================

def py_is_orthogonal(a, b, tol=1e-9):
    """Perpendicular if the dot product is ~0."""
    dot = sum(x * y for x, y in zip(a, b))
    return abs(dot) < tol


def py_project(a, b):
    """Projection of a onto b: (a·b / b·b) * b."""
    ab = sum(x * y for x, y in zip(a, b))       # how much a lies along b
    bb = sum(y * y for y in b)                  # b·b = |b|^2
    scale = ab / bb
    return [scale * y for y in b]               # times the direction b


def py_gram_schmidt(vectors):
    """Turn vectors into an orthonormal set: subtract projections, normalize."""
    result = []
    for v in vectors:
        w = list(v)
        for u in result:                        # remove overlap with earlier dirs
            p = py_project(w, u)
            w = [wi - pi for wi, pi in zip(w, p)]
        n = sum(x * x for x in w) ** 0.5        # L2 length
        result.append([x / n for x in w])       # normalize to unit length
    return result


# ======================================================================
# LEVEL 2 — NumPy
# ======================================================================

def np_is_orthogonal(a, b):
    """Perpendicular if a @ b ~ 0."""
    return bool(np.isclose(a @ b, 0.0))


def np_project(a, b):
    """Projection of a onto b."""
    return (a @ b) / (b @ b) * b


def np_gram_schmidt(vectors):
    """Orthonormal set from a list of 1-D arrays."""
    out = []
    for v in vectors:
        w = v.astype(float).copy()
        for u in out:
            w = w - (w @ u) / (u @ u) * u       # subtract projection onto u
        out.append(w / np.linalg.norm(w))       # normalize
    return out


# ======================================================================
# LEVEL 3 — PyTorch
# ======================================================================

def torch_is_orthogonal(a, b):
    """Perpendicular if dot ~ 0."""
    import torch
    return bool(torch.isclose(torch.dot(a, b), torch.tensor(0.0)))


def torch_project(a, b):
    """Projection of a onto b."""
    import torch
    return torch.dot(a, b) / torch.dot(b, b) * b


def torch_gram_schmidt(vectors):
    """Orthonormal set from a list of 1-D tensors."""
    import torch
    out = []
    for v in vectors:
        w = v.clone().float()
        for u in out:
            w = w - torch.dot(w, u) / torch.dot(u, u) * u
        out.append(w / torch.linalg.norm(w))
    return out


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

    print("\nAll tests passed!")
