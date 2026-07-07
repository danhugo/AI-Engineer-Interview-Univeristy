# Orthogonality & Projections — Q&A

---

## Orthogonality

**Q: What does it mean for two vectors to be orthogonal?**
A: They point at a right angle (90°). Their dot product is 0.

**Q: Why does orthogonal mean the dot product is 0?**
A: Because `a · b = |a| * |b| * cos(θ)`, and at 90° `cos(90°) = 0`, so the whole product is 0.

**Q: Is `[1,1]` orthogonal to `[1,0]`?**
A: No. Their dot product is `1*1 + 1*0 = 1`, which is not 0.

**Q: Why does AI care about orthogonal directions?**
A: Orthogonal directions carry independent information — no overlap or redundancy. This underlies PCA axes and orthogonal weight init.

---

## Orthonormal & Orthogonal Matrices

**Q: What is the difference between orthogonal and orthonormal?**
A: Orthogonal means perpendicular. Orthonormal means perpendicular AND each vector has length 1 (unit vectors).

**Q: What is an orthogonal matrix?**
A: A matrix `Q` whose columns are orthonormal vectors.

**Q: What are the two key properties of an orthogonal matrix?**
A: `Qᵀ Q = I`, and `Qᵀ = Q⁻¹` — the transpose is the inverse, so the inverse is free.

**Q: Why is `Qᵀ = Q⁻¹` useful?**
A: Computing an inverse is normally expensive. For an orthogonal matrix you just transpose it, which is instant.

**Q: Why do orthogonal transforms help deep networks?**
A: They keep vector lengths unchanged, so signals don't blow up or shrink. Orthogonal init keeps gradients stable.

---

## Projection

**Q: What is the projection of `a` onto `b`?**
A: The part of `a` that lies along `b` — its shadow. Formula: `proj = (a·b / b·b) * b`.

**Q: How do you read the projection formula in plain words?**
A: "How much of `a` is along `b`" (the scalar `a·b / b·b`) times the direction `b`.

**Q: What is the projection of `[3,3]` onto `[1,0]`?**
A: `[3,0]` — the shadow of `[3,3]` on the x-axis is just its x-part.

**Q: How does the formula simplify when `b` is a unit vector?**
A: `b·b = 1`, so it becomes `proj = (a · b) * b`.

**Q: Where do projections show up in AI?**
A: Least-squares fitting, attention (weighted combinations), and dropping data onto principal components.

---

## Gram-Schmidt

**Q: What does Gram-Schmidt do?**
A: It turns a set of vectors that are not perpendicular into an orthonormal set that spans the same space.

**Q: What is the core trick of Gram-Schmidt?**
A: For each new vector, subtract its projections onto the vectors already done, then normalize what's left.

**Q: Why does subtracting projections make a vector perpendicular?**
A: The projection is the overlapping part. Removing it leaves only the part that has no overlap with earlier directions — which is perpendicular.

**Q: What real algorithm uses Gram-Schmidt?**
A: QR decomposition builds its orthonormal `Q` this way.

**Q: Why can Gram-Schmidt outputs differ from an expected answer?**
A: The sign can flip — both `u` and `-u` are valid. Judge results by properties (unit length, mutually perpendicular), not exact numbers.

---

## From Scratch

**Q: How do you check orthogonality from scratch?**
A: Compute the dot product and test if its absolute value is below a small tolerance (≈ 0).

**Q: How do you code a projection from scratch?**
A: `scale = dot(a, b) / dot(b, b)`, then return `[scale * x for x in b]`.

---

## NumPy

**Q: How do you check orthogonality in NumPy?**
A: `np.isclose(a @ b, 0.0)` — dot product is `@`, compared to 0 with a tolerance.

**Q: How do you project in NumPy?**
A: `(a @ b) / (b @ b) * b`.

**Q: What NumPy function gives an orthonormal basis directly?**
A: `np.linalg.qr(A)` returns an orthonormal `Q`.

---

## PyTorch

**Q: How do you project a vector in PyTorch?**
A: `torch.dot(a, b) / torch.dot(b, b) * b`. Tensors must be float.

**Q: What PyTorch function gives an orthonormal basis?**
A: `torch.linalg.qr(A)`.

---

## Complexity

**Q: What is the cost of a projection?**
A: O(n) — one dot product and one scaling of an n-length vector.

**Q: Why is Gram-Schmidt O(k²·n) for k vectors of size n?**
A: Each vector subtracts projections onto all earlier ones, so vector `i` does about `i` projections — that sums to quadratic in `k`.

---

## Gotchas

**Q: Does orthogonal mean the vectors are unrelated?**
A: No — it specifically means perpendicular, i.e. dot product = 0.

**Q: In the projection formula, do you divide by `|b|` or `b·b`?**
A: By `b·b`, which equals `|b|²` — not `|b|`.

**Q: How should you compute the inverse of an orthogonal matrix?**
A: Don't compute it. Use the transpose `Qᵀ`, which equals the inverse for free.
