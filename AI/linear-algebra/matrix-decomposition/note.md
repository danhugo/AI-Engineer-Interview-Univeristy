# Matrix Decomposition (LU, QR, Cholesky) — Interview Knowledge Sheet

## One-Line Idea

**Decomposition = factoring a matrix into simpler pieces.** Like `12 = 4 × 3`, a matrix `A` splits into easy matrices (triangular, orthogonal). The pieces make hard problems — solving systems, least squares, sampling — fast and stable.

---

## Why Factor Instead of Invert?

To solve `A x = b`, the math textbook says `x = A⁻¹ b`. In real code you **never compute the inverse**. It is slow, uses more memory, and loses accuracy.

Instead you factor `A` once into triangular pieces, then solving becomes two quick back-substitutions. If you have many right-hand sides `b`, you factor once and reuse it. This is the whole point of decomposition.

---

## 1. LU Decomposition

Split `A` into **L**ower × **U**pper triangular:

```
A = L U

L = ones on the diagonal, values below      U = values on/above diagonal
[[1,   0],                                   [[4,  3],
 [1.5, 1]]                                    [0, -1.5]]
```

Example: `A = [[4, 3], [6, 3]]`

```
Step 1: eliminate the 6 under the 4.
  multiplier = 6 / 4 = 1.5
  row2 = row2 - 1.5 * row1  →  [6-6, 3-4.5] = [0, -1.5]

U = [[4, 3], [0, -1.5]]
L = [[1, 0], [1.5, 1]]     (the multiplier 1.5 goes in L)

Check: L @ U = [[4, 3], [6, 3]] = A  ✓
```

**Pivoting:** if a diagonal entry is 0 (or tiny), you must swap rows first. That gives `P A = L U` where `P` is a permutation (row-swap) matrix. Real libraries always pivot for stability. Our from-scratch version skips pivoting and only works on matrices that don't need it.

**Use:** solve `A x = b` fast, especially for many different `b`. Also gives the determinant (product of `U`'s diagonal).

---

## 2. QR Decomposition

Split `A` into **Q** (orthogonal) × **R** (upper triangular):

```
A = Q R
```

- `Q` is **orthogonal**: its columns are unit vectors at right angles, so `Qᵀ Q = I`. This makes `Q` numerically stable — it never stretches or shrinks.
- `R` is upper triangular.

`Q` comes from **Gram-Schmidt** on the columns of `A` — turning them into an orthonormal set (see [[orthogonality-projections]]).

**Use:** the go-to tool for **least squares** (fitting `A x ≈ b` when there is no exact answer) and as the engine inside eigenvalue algorithms. More stable than the "normal equations" `AᵀA`.

QR from scratch is fiddly (Gram-Schmidt with care for sign and near-zero columns), so we keep QR to the NumPy and PyTorch levels here.

---

## 3. Cholesky Decomposition

A special, faster LU for **symmetric positive-definite (SPD)** matrices:

```
A = L Lᵀ         (L lower triangular, A must be symmetric positive-definite)
```

Example: `A = [[4, 2], [2, 3]]` (symmetric, SPD)

```
L[0][0] = √4 = 2
L[1][0] = 2 / 2 = 1
L[1][1] = √(3 - 1²) = √2

L = [[2, 0], [1, √2]]
Check: L @ Lᵀ = [[4, 2], [2, 3]] = A  ✓
```

- **SPD** means symmetric (`A = Aᵀ`) and `xᵀ A x > 0` for any non-zero `x`. Covariance matrices are SPD.
- About **twice as fast** as LU and very stable — no pivoting needed.
- If `A` is not SPD, Cholesky fails. That failure is actually a cheap way to **test** if a matrix is SPD.

**Use:** sampling from a multivariate Gaussian (`x = mean + L z`, where `z` is standard normal), covariance work, and optimization (Newton steps).

---

## 4. From Scratch in Pure Python

LU by Gaussian elimination (no pivoting) and Cholesky by the direct formulas:

```python
def lu(A):
    n = len(A)
    U = [row[:] for row in A]                 # copy A
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for k in range(n):                        # pivot column k
        for i in range(k + 1, n):             # rows below the pivot
            m = U[i][k] / U[k][k]             # multiplier (assumes pivot != 0)
            L[i][k] = m
            for j in range(k, n):
                U[i][j] -= m * U[k][j]        # subtract to zero out below
    return L, U

def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = (A[i][i] - s) ** 0.5      # diagonal: sqrt
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]   # below diagonal
    return L
```

QR is left to the libraries (Gram-Schmidt done carefully).

---

## 5. NumPy

```python
import numpy as np

# --- QR ---
A = np.array([[1.0, 1.0], [0.0, 1.0]])
Q, R = np.linalg.qr(A)      # Q orthogonal, R upper triangular
# Q @ R == A,  Q.T @ Q == I

# --- Cholesky (A must be symmetric positive-definite) ---
A = np.array([[4.0, 2.0], [2.0, 3.0]])
L = np.linalg.cholesky(A)   # lower triangular, L @ L.T == A

# --- LU ---
# NumPy has no plain LU. Use SciPy:  from scipy.linalg import lu
# P, L, U = lu(A)   # note: SciPy returns the pivoted form P A = L U
```

`np.linalg.cholesky` returns the **lower** `L`. It raises `LinAlgError` if `A` is not SPD — a handy SPD test.

---

## 6. PyTorch

```python
import torch

# --- QR ---
A = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
Q, R = torch.linalg.qr(A)        # Q @ R == A,  Q.T @ Q == I

# --- Cholesky ---
A = torch.tensor([[4.0, 2.0], [2.0, 3.0]])
L = torch.linalg.cholesky(A)     # lower triangular, L @ L.T == A

# --- LU (with pivoting) ---
# LU_data, pivots = torch.linalg.lu_factor(A)   # for solving systems
```

Both need **float** tensors. `torch.linalg.cholesky` also returns the lower `L`.

---

## 7. Complexity

For an `n × n` matrix:

| Decomposition | Cost | Needs |
|---------------|------|-------|
| LU | O(n³) | square, (pivoting for stability) |
| QR | O(n³) | any (m×n) full-rank matrix |
| Cholesky | O(n³), ~half of LU | symmetric positive-definite |
| Solve after factoring | O(n²) per right-hand side | the factors above |

The factoring is `O(n³)` once. Each later solve is only `O(n²)` — that reuse is why we factor.

---

## 8. Interview Gotchas

| Trap | Fix |
|------|-----|
| Computing `A⁻¹` to solve `Ax=b` | Don't. Factor (LU/QR/Cholesky) and back-substitute — faster and more stable. |
| LU without pivoting | Breaks when a pivot is 0 or tiny. Real code uses `P A = L U`. |
| Comparing L/Q/R across libraries | Signs and order can differ. Verify by reconstruction (`L@U≈A`, `Q@R≈A`). |
| Cholesky on a non-SPD matrix | It errors. That error is a valid way to check "is this SPD?". |
| Forgetting `A` must be symmetric for Cholesky | Cholesky needs symmetric positive-definite, not just any matrix. |
| Using normal equations `AᵀA` for least squares | Squares the condition number (less stable). Prefer QR. |
| Passing int tensors/arrays | Decompositions need floats. |
