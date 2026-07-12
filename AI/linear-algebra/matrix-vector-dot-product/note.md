# Matrix, Vector & Dot Product — Interview Knowledge Sheet

## Intuition

A **dot product** measures how much two vectors point to the same direction.

---

## 1. Vector Dot Product

Two vectors of the **same length**. Multiply them element by element, then add up.

```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

Formula:

```
a · b = Σ aᵢ * bᵢ      (one number out, called a scalar)
```

**What it means (geometry):**

```
a · b = |a| * |b| * cos(θ)
```

- Large positive → vectors point the same way (similar).
- Zero → vectors are perpendicular (unrelated).
- Negative → vectors point opposite ways.

**Why AI cares:** dot product is the core of **similarity**. Cosine similarity, attention scores (`Q · K`), and a single neuron's output are all dot products.

---

## 2. Matrix × Vector  (`W x`)

A matrix times a vector = **one dot product per row** of the matrix.

```
W = [[1, 2, 3],        x = [1,
     [4, 5, 6]]             0,
                            1]

row 0 · x = 1*1 + 2*0 + 3*1 = 4
row 1 · x = 4*1 + 5*0 + 6*1 = 10

W x = [4, 10]
```

Shape: `(2, 3) · (3,) = (2,)`. The matrix has 2 rows → the answer has 2 numbers.

This is exactly what one layer of a neural net does to an input.

---

## 3. Matrix × Matrix  (`A B`)

Every **row of A** dotted with every **column of B**.

```
A = [[1, 2],       B = [[5, 6],
     [3, 4]]            [7, 8]]

out[0][0] = row0(A) · col0(B) = 1*5 + 2*7 = 19
out[0][1] = row0(A) · col1(B) = 1*6 + 2*8 = 22
out[1][0] = row1(A) · col0(B) = 3*5 + 4*7 = 43
out[1][1] = row1(A) · col1(B) = 3*6 + 4*8 = 50

A B = [[19, 22],
       [43, 50]]
```

Result position `[i][j]` = row `i` of A dotted with column `j` of B.

---

## 4. Shape Rules — the #1 interview bug

The **inner dimensions must match**. The outer ones become the result shape.

```
(m, n) · (n, p) = (m, p)
       └──┴── these must be equal
```

Examples:

```
(2, 3) · (3, 4) = (2, 4)   ✓
(2, 3) · (2, 4) = ERROR     ✗  (3 ≠ 2)
(2, 3) · (3,)   = (2,)      ✓  matrix × vector
```

Rule of thumb: write the shapes side by side. If the two touching numbers are equal, it works, and they cancel out.

**Matrix multiply is NOT commutative:** `A B ≠ B A` in general. Order matters.

---

## 5. Tie to Neural Networks

A **linear layer** (also called dense / fully-connected) is:

```
y = W x + b
```

- `x` — input vector (features)
- `W` — weight matrix (learned)
- `b` — bias vector (learned)
- `y` — output vector

That's it. A deep network is many of these stacked with a non-linear function between them. Understanding `W x` means you understand the core of a forward pass.

For a **batch** of inputs `X` (many rows, one per sample), you use matrix × matrix:

```
Y = X Wᵀ + b     # shape (batch, out_features)
```

---

## 6. From Scratch in Pure Python

Before the libraries, know how to write it by hand. This is what NumPy does under the hood (just much faster).

```python
def dot(a, b):
    total = 0
    for i in range(len(a)):          # same length required
        total += a[i] * b[i]
    return total

def matvec(W, x):
    return [dot(row, x) for row in W]        # one dot per row

def matmul(A, B):
    n, p = len(B), len(B[0])
    out = []
    for row in A:
        col_dots = []
        for j in range(p):
            col = [B[k][j] for k in range(n)]   # column j of B
            col_dots.append(dot(row, col))
        out.append(col_dots)
    return out
```

The three functions build on each other: `matvec` is dot per row, `matmul` is dot for every row×column pair. Slow in Python — fine for learning, never for real work.

---

## 7. NumPy

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# --- dot product (vector · vector → scalar) ---
np.dot(a, b)      # 32
a @ b             # 32   (@ is the matmul operator, cleanest)

# --- matrix × vector ---
W = np.array([[1, 2, 3],
              [4, 5, 6]])
W @ a             # array([14, 32])

# --- matrix × matrix ---
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B             # [[19, 22], [43, 50]]
np.matmul(A, B)   # same thing
```

**Watch out — `*` is NOT matmul:**

```python
A * B    # elementwise multiply: [[5,12],[21,32]]  — WRONG for matmul
A @ B    # real matrix multiply                    — RIGHT
```

**Shapes and transpose:**

```python
A.shape       # (2, 2)
A.T           # transpose — flips rows and columns
A.T.shape     # (2, 2) here; for (2,3) it becomes (3,2)
```

**Broadcasting** lets you add a bias vector to every row automatically:

```python
Y = X @ W.T + b   # b has shape (out,), added to each row of X @ W.T
```

---

## 8. PyTorch

Same ideas, `torch` tensors instead of arrays. This is what real model code uses.

```python
import torch

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

# --- dot product (1-D · 1-D → scalar) ---
torch.dot(a, b)      # tensor(32.)

# --- matrix × vector ---
W = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
torch.mv(W, a)       # tensor([14., 32.])   mv = matrix-vector

# --- matrix × matrix ---
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])
torch.matmul(A, B)   # [[19, 22], [43, 50]]
A @ B                # same — @ works on tensors too

# --- batch matrix multiply (3-D tensors) ---
torch.bmm(batchA, batchB)   # multiplies each matrix in the batch
```

`torch.matmul` and `@` are the general workhorses: they pick dot / mv / matmul / batched automatically based on the tensor shapes. `torch.dot`, `torch.mv`, `torch.bmm` are the strict, shape-specific versions.

**Shape and transpose in torch:**

```python
A.shape          # torch.Size([2, 2])
A.T              # transpose (2-D)
A.transpose(0, 1)  # swap two dims (works for any number of dims)
```

---

## 9. Complexity

| Operation | Shapes | Time |
|-----------|--------|------|
| Vector dot product | `(n) · (n)` | O(n) |
| Matrix × vector | `(m,n) · (n)` | O(m·n) |
| Matrix × matrix | `(m,n) · (n,p)` | O(m·n·p) |

Naive matmul is O(n³) for square matrices. Real libraries (NumPy, PyTorch) use tuned BLAS / GPU kernels, so they are far faster than a Python loop — always use them, never hand-roll loops.
