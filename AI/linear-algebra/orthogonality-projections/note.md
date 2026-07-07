# Orthogonality & Projections — Interview Knowledge Sheet

## One-Line Idea

Two vectors are **orthogonal** when they point at a right angle — their dot product is 0. A **projection** is the shadow one vector casts onto another.

---

## 1. Orthogonal = Perpendicular = Dot Product is 0

Two vectors are orthogonal when the angle between them is 90°.

From `a · b = |a| * |b| * cos(θ)` (see [[matrix-vector-dot-product]]), at 90° `cos(90°) = 0`, so:

```
a ⊥ b   ⟺   a · b = 0
```

```
a = [1, 0]      b = [0, 1]
a · b = 1*0 + 0*1 = 0   → orthogonal ✓

a = [1, 1]      b = [1, 0]
a · b = 1*1 + 1*0 = 1   → NOT orthogonal ✗
```

**Why AI cares:** orthogonal directions carry independent information — no overlap, no redundancy. This is the idea behind PCA axes and orthogonal weight initialization.

---

## 2. Orthonormal & Orthogonal Matrices

- **Orthonormal** = orthogonal **and** each vector has length 1 (unit vectors). "Ortho" (perpendicular) + "normal" (unit length).
- An **orthogonal matrix** `Q` has orthonormal columns.

An orthogonal matrix has two magic properties:

```
Qᵀ Q = I          (columns are perpendicular unit vectors)
Qᵀ = Q⁻¹          (the transpose IS the inverse — free, no computing)
```

The second one is huge: normally finding an inverse is expensive, but for `Q` you just flip it.

**Why AI cares:** orthogonal transforms keep vector lengths unchanged (no blowing up or shrinking). That is why orthogonal init keeps gradients stable in deep nets, and why QR/SVD lean on orthonormal bases.

---

## 3. Projection of One Vector onto Another

The projection of `a` onto `b` is the piece of `a` that lies along `b` — its shadow.

```
proj_b(a) = (a · b / b · b) * b
```

Read it as: "how much of `a` is along `b`" (the scalar `a·b / b·b`) times the direction `b`.

```
a = [3, 3]      b = [1, 0]   (the x-axis)

a·b = 3,  b·b = 1
proj = (3 / 1) * [1, 0] = [3, 0]
```

The shadow of `[3,3]` on the x-axis is `[3,0]` — just the x-part. Makes sense.

If `b` is a **unit** vector, `b·b = 1`, so it simplifies to `proj = (a · b) * b`.

**Why AI cares:** projection is the core of least-squares fitting (project data onto the model space), attention (weighted combinations), and dropping data onto principal components.

---

## 4. Gram-Schmidt — Build an Orthonormal Set

Given a few vectors that are NOT perpendicular, Gram-Schmidt turns them into an orthonormal set that spans the same space.

The trick: for each new vector, **subtract its projections onto the ones already done**, then normalize what's left.

```
Steps for vectors v1, v2, ...:
  u1 = normalize(v1)
  u2 = v2 - proj_u1(v2)   → then normalize
  u3 = v3 - proj_u1(v3) - proj_u2(v3)   → then normalize
  ...
```

Each subtraction removes the part that overlaps with earlier directions, leaving something perpendicular.

**Why AI cares:** this is exactly how **QR decomposition** builds its orthonormal `Q` (link `[[matrix-decomposition]]`). Orthonormal bases make many problems numerically stable and easy to solve.

Note: the **sign** of a Gram-Schmidt result can flip depending on the exact steps — both `u` and `-u` are valid. Judge results by their *properties* (unit length, mutually perpendicular), not exact numbers.

---

## 5. From Scratch in Pure Python

```python
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def is_orthogonal(a, b, tol=1e-9):
    return abs(dot(a, b)) < tol           # dot ≈ 0 means perpendicular

def project(a, b):
    scale = dot(a, b) / dot(b, b)         # how much of a lies along b
    return [scale * x for x in b]         # times the direction b

def normalize(v):
    n = dot(v, v) ** 0.5
    return [x / n for x in v]

def gram_schmidt(vectors):
    result = []
    for v in vectors:
        w = list(v)
        for u in result:                  # subtract projections onto done ones
            p = project(w, u)
            w = [wi - pi for wi, pi in zip(w, p)]
        result.append(normalize(w))
    return result
```

---

## 6. NumPy

```python
import numpy as np

a = np.array([3.0, 3.0])
b = np.array([1.0, 0.0])

# orthogonal check
np.isclose(a @ b, 0.0)                    # dot product ≈ 0 ?

# projection of a onto b
proj = (a @ b) / (b @ b) * b              # [3., 0.]

# normalize
b_hat = b / np.linalg.norm(b)

# Gram-Schmidt (loop version — clear over clever)
def gram_schmidt(vectors):
    out = []
    for v in vectors:
        w = v.astype(float).copy()
        for u in out:
            w = w - (w @ u) / (u @ u) * u
        out.append(w / np.linalg.norm(w))
    return out
```

NumPy also has `np.linalg.qr(A)` which returns an orthonormal `Q` directly — the production way to get an orthonormal basis.

---

## 7. PyTorch

```python
import torch

a = torch.tensor([3.0, 3.0])
b = torch.tensor([1.0, 0.0])

# orthogonal check
torch.isclose(torch.dot(a, b), torch.tensor(0.0))

# projection of a onto b
proj = torch.dot(a, b) / torch.dot(b, b) * b     # tensor([3., 0.])

# normalize
b_hat = b / torch.linalg.norm(b)

# Gram-Schmidt (loop version)
def gram_schmidt(vectors):
    out = []
    for v in vectors:
        w = v.clone().float()
        for u in out:
            w = w - torch.dot(w, u) / torch.dot(u, u) * u
        out.append(w / torch.linalg.norm(w))
    return out
```

PyTorch also has `torch.linalg.qr(A)` for a ready-made orthonormal `Q`. Tensors must be float.

---

## 8. Complexity

| Operation | Work |
|-----------|------|
| Orthogonal check (dot) | O(n) |
| Projection | O(n) — one dot, one scale |
| Normalize | O(n) |
| Gram-Schmidt (k vectors, size n) | O(k²·n) — each vector subtracts all earlier ones |

Gram-Schmidt is quadratic in the number of vectors because vector `i` subtracts projections onto all `i-1` earlier ones.

---

## 9. Interview Gotchas

| Trap | Fix |
|------|-----|
| "Orthogonal" means unrelated? | It means perpendicular: dot product = 0. |
| Orthogonal vs orthonormal | Orthonormal = orthogonal **plus** unit length. |
| Dividing by `b·b` vs `|b|` in projection | Formula uses `b·b`. It's `|b|²`, not `|b|`. |
| Expecting exact Gram-Schmidt output | Sign can flip. Check properties: unit length + mutually perpendicular. |
| Computing `Q⁻¹` for an orthogonal matrix | Don't — just use `Qᵀ`. It's the inverse for free. |
| Projecting onto a zero vector | `b·b = 0` → divide by zero. Guard against it. |
| Int tensors in torch | Norms and division need floats. Use float tensors. |
