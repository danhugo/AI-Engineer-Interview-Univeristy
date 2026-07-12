# Matrix Transpose, Inverse & Multiply — Interview Knowledge Sheet

## Intuition

**Transpose** flips a matrix over its diagonal. **Inverse** is the "undo" of a matrix — like `1/x` but for matrices. **Multiply** chains two transforms into one.

---

## 1. Transpose

Swap rows and columns. Position `[i][j]` becomes `[j][i]`.

```
A = [[1, 2, 3],          Aᵀ = [[1, 4],
     [4, 5, 6]]                [2, 5],
                               [3, 6]]
```

Shape flips too: `(2, 3)` becomes `(3, 2)`.

**Why AI cares:** you transpose weight matrices all the time. A batch linear layer is `X @ W.T`. Also `AᵀA` shows up in least squares (see below).

Rule: `(Aᵀ)ᵀ = A` (transpose twice gets you back). And `(AB)ᵀ = BᵀAᵀ` (order flips).

---

## 2. Matrix Multiply (recap)

Every **row of A** dotted with every **column of B**. Output `[i][j]` = row `i` of A · column `j` of B.

```
A = [[1, 2],       B = [[5, 6],
     [3, 4]]            [7, 8]]

A B = [[1*5+2*7, 1*6+2*8],   =  [[19, 22],
       [3*5+4*7, 3*6+4*8]]       [43, 50]]
```

Shape rule: `(m,n) · (n,p) = (m,p)` — inner dims must match. Not commutative: `AB ≠ BA`.

---

## 3. Determinant

One number that tells you if a matrix can be inverted. For a 2×2:

```
A = [[a, b],        det(A) = a*d - b*c
     [c, d]]

A = [[4, 7],        det = 4*6 - 7*2 = 24 - 14 = 10
     [2, 6]]
```

**Key fact:** if `det = 0`, the matrix has **no inverse** (it is "singular"). Think of it as the matrix squashing space flat — you can't un-squash it.

Geometrically, `|det|` is the area (2D) or volume (3D) that the matrix scales space by. A zero determinant means it collapsed a dimension.

---

## 4. Inverse

The inverse `A⁻¹` undoes A: `A @ A⁻¹ = I` (the identity matrix). Identity is the "1" of matrices — all 1s on the diagonal, 0s elsewhere.

```
I = [[1, 0],
     [0, 1]]
```

**2×2 formula:** swap the diagonal, negate the off-diagonal, divide everything by the determinant.

```
A = [[a, b],        A⁻¹ = (1/det) * [[ d, -b],
     [c, d]]                        [-c,  a]]

A = [[4, 7],   det = 10
     [2, 6]]

A⁻¹ = (1/10) * [[ 6, -7],   =  [[ 0.6, -0.7],
                [-2,  4]]       [-0.2,  0.4]]
```

Check: `A @ A⁻¹` should give the identity.

---

## 5. Why the Inverse Matters — Solving Linear Systems

A system of equations `A x = b` (find `x`) has the clean solution:

```
x = A⁻¹ b
```

Multiply both sides by `A⁻¹` on the left: `A⁻¹ A x = A⁻¹ b` → `I x = A⁻¹ b` → `x = A⁻¹ b`.

**AI tie-in — least squares / normal equations.** To fit a line or a linear model, you solve:

```
w = (Xᵀ X)⁻¹ Xᵀ y
```

This is linear regression in closed form. Transpose, multiply, and inverse all show up in one line — that is why these three operations travel together.

---

## 6. When the Inverse Does NOT Exist

An inverse exists only if `det ≠ 0`. If `det = 0` the matrix is **singular** — its rows/columns are linearly dependent (one is a combo of the others), so information is lost and you can't undo it.

```
[[1, 2],       det = 1*4 - 2*2 = 0   → NO inverse
 [2, 4]]       (row 2 is just 2× row 1)
```

Only **square** matrices can have an inverse. Non-square matrices use the pseudo-inverse instead.

---

## 7. From Scratch in Pure Python

```python
def transpose(A):
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]

def matmul(A, B):
    n, p = len(B), len(B[0])
    out = []
    for row in A:
        out.append([sum(row[k] * B[k][j] for k in range(n)) for j in range(p)])
    return out

def det2(A):
    (a, b), (c, d) = A         # unpack a 2x2
    return a * d - b * c

def inv2(A):
    (a, b), (c, d) = A
    det = a * d - b * c
    if det == 0:
        raise ValueError("singular matrix — no inverse")
    return [[ d / det, -b / det],
            [-c / det,  a / det]]
```

`det2`/`inv2` only handle 2×2. General N×N determinant/inverse is a lot more code (Gaussian elimination) — in real work you always use a library.

---

## 8. NumPy

```python
import numpy as np

A = np.array([[4., 7.],
              [2., 6.]])

A.T                     # transpose → [[4,2],[7,6]]
A @ B                   # matrix multiply
np.linalg.det(A)        # 10.0
np.linalg.inv(A)        # [[0.6,-0.7],[-0.2,0.4]]

A @ np.linalg.inv(A)    # ≈ identity [[1,0],[0,1]]
np.eye(2)               # the 2x2 identity matrix

# Solving A x = b — DO THIS, not inv:
x = np.linalg.solve(A, b)   # faster and more stable than inv(A) @ b
```

---

## 9. PyTorch

```python
import torch

A = torch.tensor([[4., 7.],
                  [2., 6.]])

A.T                       # transpose (2-D)
A @ B                     # matrix multiply
torch.linalg.det(A)       # tensor(10.)
torch.linalg.inv(A)       # [[0.6,-0.7],[-0.2,0.4]]

A @ torch.linalg.inv(A)   # ≈ identity
torch.eye(2)              # 2x2 identity

# Solving A x = b — preferred over inv:
x = torch.linalg.solve(A, b)
```

`torch.linalg.det` / `inv` / `solve` need float tensors.

---

## 10. Complexity

| Operation | Shapes | Time |
|-----------|--------|------|
| Transpose | `(m,n)` | O(m·n) — or O(1) as a "view" (no data moved) |
| Multiply | `(m,n)·(n,p)` | O(m·n·p) |
| Determinant (N×N) | `(n,n)` | O(n³) |
| Inverse (N×N) | `(n,n)` | O(n³) |
| Solve `Ax=b` | `(n,n)` | O(n³) but smaller constant than inv |

NumPy/PyTorch often return a transpose as a **view** — they just relabel axes without copying, so it is effectively free.

---

## 11. Interview Gotchas

| Trap | Fix |
|------|-----|
| Using `inv(A) @ b` to solve a system | Use `solve(A, b)` — faster and more numerically stable. |
| Inverting a singular matrix | Check `det ≠ 0` first, or catch the error. `det=0` → no inverse. |
| Assuming any matrix has an inverse | Only square matrices with `det ≠ 0`. Use pseudo-inverse otherwise. |
| `(AB)ᵀ = AᵀBᵀ` | Wrong — the order flips: `(AB)ᵀ = BᵀAᵀ`. |
| Comparing floats with `==` | Inverses have rounding error. Use a tolerance (`allclose`). |
| `ord=2` confusion vs transpose | Transpose is `.T`; it has nothing to do with norms. |
