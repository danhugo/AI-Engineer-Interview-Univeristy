# SVD (Singular Value Decomposition) — Interview Knowledge Sheet

## One-Line Idea

**Any** matrix, no matter how messy, is really just three simple steps: **rotate → stretch → rotate**. SVD pulls those three steps apart so you can see them.

---

## 1. The Decomposition

Every matrix `A` (any shape, `m × n`) can be split into three matrices:

```
A = U Σ Vᵀ
```

- `Vᵀ` — a rotation (orthogonal matrix). Turns the input.
- `Σ` (sigma) — a diagonal matrix. Stretches along each axis. The diagonal values are the **singular values**.
- `U` — another rotation (orthogonal matrix). Turns the output.

Shapes: `A (m×n) = U (m×m) · Σ (m×n) · Vᵀ (n×n)`.

"Orthogonal" means the columns are perpendicular unit vectors — the matrix only rotates/reflects, it never stretches. All the stretching lives in `Σ`.

---

## 2. Singular Values

The diagonal of `Σ` holds the **singular values** `σ₁, σ₂, …`.

- They are always **≥ 0**.
- They are sorted **largest first**: `σ₁ ≥ σ₂ ≥ … ≥ 0`.
- Each one says **how much the matrix stretches** in one direction.

Big singular value = important direction (lots of stretch). Tiny singular value = weak direction (barely matters — often just noise).

```
A = [[3, 0],
     [0, 2]]

singular values = [3, 2]   (stretches x by 3, y by 2)
```

---

## 3. Geometric Meaning

Take the unit circle and apply `A`. You get an ellipse. SVD explains the ellipse:

```
Vᵀ : rotate the circle
Σ  : stretch it into an ellipse (axis lengths = singular values)
U  : rotate the ellipse to its final place
```

The singular values are the **lengths of the ellipse's axes**. That is the whole picture.

---

## 4. Link to Eigenvalues

SVD is closely tied to eigenvalues (see [[eigenvalues-eigenvectors]]).

**Key fact:** the singular values of `A` are the **square roots of the eigenvalues of `AᵀA`**.

```
σᵢ = √(λᵢ of AᵀA)
```

Why `AᵀA`? Because `AᵀA` is always symmetric and its eigenvalues are ≥ 0, so their square roots are real and non-negative — exactly what singular values need to be. This is how you can find singular values by hand for a small matrix.

- The columns of `V` are the eigenvectors of `AᵀA`.
- The columns of `U` are the eigenvectors of `AAᵀ`.

---

## 5. Why AI Cares

SVD shows up everywhere:

| Use | Idea |
|-----|------|
| **Low-rank approximation / compression** | Keep only the top-k singular values, drop the tiny ones. Rebuild a smaller, almost-identical matrix. |
| **PCA** | Principal Component Analysis is SVD on centered data. Top singular directions = main patterns. |
| **LoRA** | Fine-tuning trick: a weight update is stored as a low-rank product (few singular directions), so it's tiny. |
| **Noise reduction** | Small singular values usually = noise. Zero them out to clean data. |
| **Pseudo-inverse** | Solve `Ax = b` even when `A` isn't square/invertible, using `1/σᵢ`. |

The core trick: **most of a matrix's "energy" lives in a few big singular values.** Keep those, throw away the rest.

---

## 6. Rank-k Approximation

Sum only the top-k singular pieces:

```
Aₖ = Σ (σᵢ · uᵢ · vᵢᵀ)   for i = 1..k
```

- `k = full rank` → you get `A` back exactly.
- Smaller `k` → smaller storage, small error.

Storing an `m×n` matrix costs `m·n` numbers. Rank-k costs `k·(m + n + 1)`. For big matrices and small `k`, that's a huge saving — this is image/model compression.

---

## 7. From Scratch in Pure Python

Full SVD by hand is hard. But two pieces are easy and show the idea.

**(a) Singular values via `AᵀA`** — works cleanly for a 2×2 matrix using the quadratic formula for a symmetric matrix's eigenvalues.

```python
def transpose(A):
    return [[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]

def matmul(A, B):
    n, p = len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
            for i in range(len(A))]

def singular_values_via_ata(A):
    S = matmul(transpose(A), A)          # AᵀA is symmetric
    a, b, c = S[0][0], S[0][1], S[1][1]  # [[a, b], [b, c]]
    tr, det = a + c, a * c - b * b       # eigenvalues of 2x2 symmetric:
    disc = (tr * tr - 4 * det) ** 0.5    #   λ = (tr ± √(tr²-4det)) / 2
    l1, l2 = (tr + disc) / 2, (tr - disc) / 2
    vals = [max(l, 0) ** 0.5 for l in (l1, l2)]  # σ = √λ, clamp tiny negatives
    return sorted(vals, reverse=True)    # largest first
```

**(b) Reconstruct** — rebuild `A` from `U, s, Vᵀ` to prove `A = U Σ Vᵀ`.

```python
def reconstruct(U, s, Vt):
    Sigma = [[s[i] if i == j else 0 for j in range(len(Vt))]
             for i in range(len(U))]
    return matmul(matmul(U, Sigma), Vt)   # U @ diag(s) @ Vt
```

You don't compute `U` and `V` by hand — you let a library do that, then check the pieces multiply back to `A`.

---

## 8. NumPy

```python
import numpy as np

A = np.array([[3.0, 0.0],
              [0.0, 2.0]])

U, s, Vt = np.linalg.svd(A)
# U   → (m, m) orthogonal
# s   → 1-D array of singular values, sorted descending  → [3., 2.]
# Vt  → (n, n) already transposed (it IS Vᵀ, not V)

# rebuild A:  U @ diag(s) @ Vt
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, s)
A_rebuilt = U @ Sigma @ Vt        # ≈ A

# rank-k approximation (keep top k singular values)
k = 1
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

Key gotchas:
- `s` is a **1-D array**, not a full diagonal matrix. Build the diagonal yourself to reconstruct.
- NumPy returns **`Vt` (already transposed)**, not `V`. Don't transpose it again.
- Use `full_matrices=False` for a thin SVD on tall/wide matrices (smaller `U`, `Vt`).

---

## 9. PyTorch

```python
import torch

A = torch.tensor([[3.0, 0.0],
                  [0.0, 2.0]])

U, S, Vh = torch.linalg.svd(A)
# same layout as numpy:  Vh is Vᵀ, S is 1-D singular values (descending)

# rebuild:
A_rebuilt = U @ torch.diag(S) @ Vh

# thin SVD:
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
```

`torch.linalg.svd` needs a **float** tensor. `torch.svd` is the old API (returns `V`, not `Vᵀ`) — prefer `torch.linalg.svd` to match NumPy.

---

## 10. Complexity

| Operation | Cost |
|-----------|------|
| Full SVD of `m×n` | O(min(m,n) · m · n) — expensive |
| Rank-k (truncated) SVD | Much cheaper — only computes top-k |
| Reconstruct from `U, s, Vᵀ` | O(m · n · rank) |

SVD is costly, so for huge matrices people use **truncated** SVD (only the top-k), never the full thing.

---

## 11. Interview Gotchas

| Trap | Fix |
|------|-----|
| Comparing `U`/`V` from two SVDs | Columns have a **sign flip** freedom — `u` and `−u` are both valid. Only singular values are unique. Compare `s`, or check reconstruction. |
| Thinking `s` is a matrix | `s` is a 1-D list of values. Put them on a diagonal to use them. |
| Transposing `Vt` again | NumPy/torch already return `Vᵀ`. Transposing gives `V` and breaks the rebuild. |
| Singular values vs eigenvalues | Not the same. `σ = √(eigenvalue of AᵀA)`. Singular values are always ≥ 0; eigenvalues can be negative. |
| Forgetting to sort | Singular values come sorted descending. Rely on that for "top-k". |
| Int tensor into torch SVD | Needs floats. Convert first. |
