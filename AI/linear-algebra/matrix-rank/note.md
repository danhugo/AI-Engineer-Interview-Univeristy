# Rank of a Matrix — Interview Knowledge Sheet

## Intuition

The **rank** is how many rows (or columns) carry real, non-redundant information. If one row is just a copy or a scaled/added version of others, it adds nothing — it doesn't count.

---

## 1. What Rank Means

Rank = the number of **linearly independent** rows.

"Linearly independent" = no row can be built from the others by scaling and adding.

```
M = [[1, 2],
     [2, 4]]      ← row 2 is exactly 2 × row 1
```

Row 2 is redundant. Only 1 row carries new information → **rank 1**.

```
M = [[1, 2],
     [3, 4]]      ← row 2 is NOT a multiple of row 1
```

Both rows are independent → **rank 2**.

**Key fact:** row rank always equals column rank. So counting independent rows gives the same number as counting independent columns. You can use whichever is easier.

---

## 2. Full Rank vs Rank-Deficient

For an `m × n` matrix, the max possible rank is `min(m, n)`.

- **Full rank** — rank equals `min(m, n)`. Every row (or column) is independent.
- **Rank-deficient** — rank is less than `min(m, n)`. Some rows are redundant.

For a **square** matrix (`n × n`):

| Rank | Meaning |
|------|---------|
| Full (rank = n) | Invertible. Determinant ≠ 0. |
| Deficient (rank < n) | Singular. Determinant = 0. No inverse. |

So "full rank", "invertible", and "det ≠ 0" all say the same thing for a square matrix.

---

## 3. How to Find Rank

**Method 1 — Row reduction (Gaussian elimination):**
1. Use row operations to turn the matrix into an upper-triangular / staircase form.
2. Count the rows that are **not all zero**. That count is the rank.

Redundant rows collapse to all-zeros during elimination, so they don't get counted.

**Method 2 — Singular values:**
Rank = number of **non-zero singular values** (see `[[svd]]`). This is what libraries actually do — it is more stable with floating-point numbers than raw elimination. Values below a tiny tolerance are treated as zero.

---

## 4. Why AI Cares

- **Collinear features** — if two input features are copies or scaled versions of each other, the data matrix is rank-deficient. This breaks linear regression (no unique solution).
- **Low-rank structure** — big matrices often carry little real information; a low-rank approximation compresses them (image compression, recommendations).
- **LoRA** — fine-tunes large models by adding a small **low-rank** update instead of changing all weights. Cheap because low rank = few real parameters.
- **Degenerate systems** — a rank-deficient system of equations has either no solution or infinitely many, never exactly one.

---

## 5. From Scratch in Pure Python

Gaussian elimination, then count non-zero rows. Use a small tolerance so tiny float noise counts as zero.

```python
def rank_by_elimination(M, tol=1e-9):
    # work on a float copy so we don't change the input
    A = [[float(x) for x in row] for row in M]
    rows, cols = len(A), len(A[0])
    rank = 0
    row = 0                          # next row to place a pivot in

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
            continue                 # no pivot here, try the next column

        A[row], A[pivot] = A[pivot], A[row]     # swap pivot row up
        # eliminate this column from all rows below
        for i in range(row + 1, rows):
            factor = A[i][col] / A[row][col]
            for j in range(col, cols):
                A[i][j] -= factor * A[row][j]
        rank += 1
        row += 1

    return rank
```

The idea: each time we lock in a pivot, that's one independent row → `rank += 1`. Redundant rows become all-zero and never provide a pivot.

---

## 6. NumPy

```python
import numpy as np

M = np.array([[1, 2],
              [2, 4]])
np.linalg.matrix_rank(M)      # 1  (rows are dependent)

np.linalg.matrix_rank(np.array([[1, 2], [3, 4]]))   # 2  (full rank)
```

`np.linalg.matrix_rank` counts singular values above a tolerance. It handles float noise for you — safer than hand-rolled elimination.

---

## 7. PyTorch

```python
import torch

M = torch.tensor([[1., 2.],
                  [2., 4.]])
torch.linalg.matrix_rank(M)   # tensor(1)

torch.linalg.matrix_rank(torch.tensor([[1., 3.], [2., 4.]]))  # tensor(2)
```

Same idea as NumPy: it uses singular values under the hood. Needs a float tensor.

---

## 8. Complexity

| Task | Work |
|------|------|
| Rank by elimination | O(m · n · min(m,n)) — roughly one pass per pivot |
| Rank by SVD (libraries) | O(m · n · min(m,n)) — same order, but numerically stable |

For a square `n × n` matrix both are about O(n³).

---

## 9. Interview Gotchas

| Trap | Fix |
|------|-----|
| Counting rows before reducing | Redundant rows look independent until elimination collapses them. Reduce first. |
| Row rank ≠ column rank? | They are always equal. Count whichever is easier. |
| "det = 0" meaning | For a square matrix, det = 0 ⇔ rank-deficient ⇔ not invertible. |
| Using exact `== 0` on floats | Float math leaves tiny noise. Use a tolerance (e.g. 1e-9). |
| Max rank confusion | Max rank is `min(m, n)`, not `max(m, n)`. |
| Thinking more rows = more rank | A 100×2 matrix has rank at most 2. |
