# Vector & Matrix Norms — Interview Knowledge Sheet

## One-Line Idea

A **norm** is the *size* (length) of a vector — one number that says "how big" it is. Different norms measure "big" in different ways.

---

## 1. L2 Norm (Euclidean) — the default

The straight-line length. Square each element, add them, take the square root.

```
v = [3, -4]

||v||₂ = √(3² + (-4)²) = √(9 + 16) = √25 = 5
```

Formula:

```
||v||₂ = √(Σ vᵢ²)
```

This is the "length" you learned in geometry. When someone says "the norm" with no other word, they mean L2.

**Why AI cares:** L2 is used in weight decay (L2 regularization), distance between embeddings, and gradient clipping.

---

## 2. L1 Norm (Manhattan)

Add up the absolute values. Like walking city blocks — you can't cut diagonally.

```
v = [3, -4]

||v||₁ = |3| + |-4| = 3 + 4 = 7
```

Formula:

```
||v||₁ = Σ |vᵢ|
```

**Why AI cares:** L1 regularization pushes weights to exactly zero, so it makes models **sparse** (many zeros). Good for feature selection.

---

## 3. L-infinity Norm (Max)

Just the largest absolute value. Ignores everything else.

```
v = [3, -4]

||v||∞ = max(|3|, |-4|) = 4
```

Formula:

```
||v||∞ = max |vᵢ|
```

**Why AI cares:** used in gradient clipping by max value and in some robustness bounds.

---

## 4. General Lp Norm

L1, L2, L-inf are all special cases of one formula:

```
||v||ₚ = (Σ |vᵢ|ᵖ)^(1/p)
```

- p = 1 → L1 (sum of absolutes)
- p = 2 → L2 (Euclidean)
- p → ∞ → L-infinity (max)

Bigger p puts more weight on the largest element.

---

## 5. Frobenius Norm — L2 for a matrix

The L2 norm, but over **all entries of a matrix** as if it were one long vector. Square every entry, add, square-root.

```
M = [[1, 2],
     [3, 4]]

||M||_F = √(1² + 2² + 3² + 4²) = √30 ≈ 5.477
```

Formula:

```
||M||_F = √(Σᵢ Σⱼ Mᵢⱼ²)
```

**Why AI cares:** measures the overall size of a weight matrix — used in regularization and to check how much weights changed during training.

---

## 6. Normalization — make length 1

Divide a vector by its L2 norm. The result points the same way but has length 1 (a **unit vector**).

```
v = [3, -4]      ||v||₂ = 5

v_normalized = v / 5 = [0.6, -0.8]      (length is now 1)
```

Formula:

```
v̂ = v / ||v||₂
```

**Why AI cares:** normalized vectors make **cosine similarity** just a dot product (see [[matrix-vector-dot-product]]). Embeddings are often normalized so only direction matters, not magnitude.

Watch out: if `||v|| = 0` (all zeros), dividing blows up. Guard against it.

---

## 7. From Scratch in Pure Python

```python
def l1_norm(v):
    return sum(abs(x) for x in v)

def l2_norm(v):
    return sum(x * x for x in v) ** 0.5      # sqrt via ** 0.5

def linf_norm(v):
    return max(abs(x) for x in v)

def frobenius_norm(M):
    return sum(x * x for row in M for x in row) ** 0.5

def normalize(v):
    n = l2_norm(v)
    return [x / n for x in v]                 # assumes n != 0
```

Frobenius is just L2 over every entry — note the double loop flattening the matrix.

---

## 8. NumPy

Use `np.linalg.norm`. The `ord` argument picks which norm.

```python
import numpy as np

v = np.array([3.0, -4.0])

np.linalg.norm(v)          # 5.0    — L2 is the default
np.linalg.norm(v, ord=1)   # 7.0    — L1
np.linalg.norm(v, ord=2)   # 5.0    — L2
np.linalg.norm(v, ord=np.inf)  # 4.0 — L-infinity

# Frobenius norm of a matrix
M = np.array([[1., 2.], [3., 4.]])
np.linalg.norm(M)              # 5.477  — 'fro' is default for a 2-D array
np.linalg.norm(M, ord='fro')  # same

# normalize
v_hat = v / np.linalg.norm(v)   # [0.6, -0.8]
```

Careful: for a 2-D array, `ord=1` and `ord=2` mean *matrix* norms (not what you expect). Use `ord='fro'` for Frobenius. To get per-row or per-column norms, use `axis`:

```python
np.linalg.norm(M, axis=1)   # L2 norm of each row
```

---

## 9. PyTorch

Use `torch.linalg.norm` (or `torch.norm`, older).

```python
import torch

v = torch.tensor([3.0, -4.0])

torch.linalg.norm(v)              # tensor(5.)    — L2 default
torch.linalg.norm(v, ord=1)       # tensor(7.)    — L1
torch.linalg.norm(v, ord=float('inf'))  # tensor(4.) — L-inf

# Frobenius norm of a matrix
M = torch.tensor([[1., 2.], [3., 4.]])
torch.linalg.norm(M)              # tensor(5.477)  — Frobenius default for 2-D
torch.linalg.norm(M, ord='fro')   # same

# normalize (built-in helper)
import torch.nn.functional as F
F.normalize(v, dim=0)             # unit vector, L2 by default
```

`torch.linalg.norm` needs a float tensor. `dim=` works like NumPy's `axis=` for per-row / per-column norms.

---

## 10. Complexity

| Norm | Work |
|------|------|
| L1 | O(n) — one pass, sum absolutes |
| L2 | O(n) — one pass, sum squares + one sqrt |
| L-infinity | O(n) — one pass, track max |
| Frobenius | O(m·n) — every matrix entry once |
| Normalize | O(n) — compute L2, then divide |

All are a single pass over the data — cheap.

---

## 11. Interview Gotchas

| Trap | Fix |
|------|-----|
| "The norm" is ambiguous | Default is L2, but say which one. |
| Normalizing a zero vector | Dividing by 0 → NaN/inf. Guard: skip or add a tiny epsilon. |
| `ord=2` on a 2-D array in NumPy | That's the matrix 2-norm (largest singular value), NOT Frobenius. Use `ord='fro'`. |
| Forgetting the square root in L2 | `Σ vᵢ²` alone is the *squared* norm. Take sqrt for the real norm. |
| L1 vs L2 regularization mix-up | L1 → sparse weights (zeros). L2 → small but non-zero weights. |
| Passing an int tensor to torch norm | Norms need floats. Use float tensors. |
