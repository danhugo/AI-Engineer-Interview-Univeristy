# Eigenvalues & Eigenvectors — Q&A

---

## Definition

**Q: What is an eigenvector?**
A: A non-zero vector whose direction does not change when you multiply it by a matrix. The matrix only stretches or shrinks it.

**Q: What is an eigenvalue?**
A: The scale factor `λ`. It says how much the eigenvector gets stretched. If `A v = λ v`, then `λ` is the eigenvalue for eigenvector `v`.

**Q: What does the equation `A v = λ v` mean in words?**
A: Multiplying the vector `v` by the matrix `A` gives back the same `v`, just scaled by the number `λ`.

---

## Geometric Meaning

**Q: What makes an eigenvector special geometrically?**
A: Most vectors get rotated and stretched by a matrix. An eigenvector only gets stretched (or flipped/shrunk) — its direction stays the same.

**Q: What do different eigenvalue signs mean?**
A: `λ > 1` stretches longer, `0 < λ < 1` shrinks, `λ < 0` flips to the opposite direction, `λ = 0` collapses the vector to zero.

**Q: How many eigenvalues does an n×n matrix have?**
A: Up to `n` of them.

---

## Characteristic Equation

**Q: What is the characteristic equation?**
A: `det(A − λI) = 0`. Solving it gives the eigenvalues. It comes from requiring `(A − λI)v = 0` to have a non-zero solution.

**Q: Why must `det(A − λI)` be zero?**
A: For a non-zero eigenvector to exist, the matrix `(A − λI)` must be singular. A singular matrix has determinant zero.

**Q: What is the characteristic equation for a 2×2 matrix?**
A: `λ² − trace·λ + det = 0`, where trace is `a + d` (diagonal sum) and det is `a·d − b·c`.

**Q: How do you solve the 2×2 characteristic equation?**
A: The quadratic formula: `λ = (trace ± √(trace² − 4·det)) / 2`.

**Q: For `A = [[2, 1], [1, 2]]`, what are the eigenvalues?**
A: Trace = 4, det = 3, so `λ = (4 ± √(16−12))/2 = (4 ± 2)/2`, giving `λ = 3` and `λ = 1`.

---

## Why AI Cares

**Q: How are eigenvectors used in PCA?**
A: The eigenvectors of the data's covariance matrix are the principal directions (axes with the most spread). The eigenvalues say how much variance each axis holds. You keep the top few to reduce dimensions.

**Q: What do eigenvalues tell you about stability?**
A: They tell you if a system grows or settles — for example, whether gradients explode or shrink.

**Q: What is one famous use of the dominant eigenvector?**
A: PageRank — the ranking vector is the dominant eigenvector of the web link matrix.

---

## Power Iteration

**Q: What does power iteration find?**
A: The dominant eigenvector — the one with the largest `|λ|`. Not all of them, just the biggest.

**Q: How does power iteration work?**
A: Start with any vector, multiply by `A`, normalize, and repeat. The vector slowly turns toward the dominant eigenvector because each multiply amplifies that direction the most.

**Q: How do you get the eigenvalue from power iteration?**
A: Use the Rayleigh quotient: `λ = (vᵀ A v) / (vᵀ v)`. If `v` is unit length, it is just `vᵀ A v`.

---

## From Scratch

**Q: How do you code the eigenvalues of a 2×2 matrix from scratch?**
A: Compute trace and det, then apply the quadratic formula. `disc = √(trace² − 4·det)`, and the two eigenvalues are `(trace ± disc)/2`.

**Q: What is the core loop of power iteration in code?**
A: Repeatedly set `v = normalize(A @ v)`. After the loop, the eigenvalue is `dot(v, A @ v)` when `v` is unit length.

---

## NumPy

**Q: How do you get eigenvalues in NumPy?**
A: `np.linalg.eigvals(A)` for values only, or `np.linalg.eig(A)` for both values and vectors.

**Q: In `vals, vecs = np.linalg.eig(A)`, where is the eigenvector?**
A: `vecs[:, i]` — it is the `i`-th column, and it is already unit length. Eigenvectors are columns, not rows.

**Q: When should you use `np.linalg.eigh`?**
A: For symmetric matrices. It is faster, returns real results, and gives them sorted.

---

## PyTorch

**Q: How do you get eigenvalues and eigenvectors in PyTorch?**
A: `torch.linalg.eig(A)` for both, or `torch.linalg.eigvals(A)` for values only.

**Q: What dtype does `torch.linalg.eig` return?**
A: Complex — even for real matrices. Take `.real` if you know the eigenvalues are real, or use `eigh` for symmetric matrices.

---

## Complexity

**Q: What is the cost of a full eigen-decomposition?**
A: O(n³), about the same as matrix multiply.

**Q: Why is power iteration cheaper?**
A: One step is a matrix-vector multiply, O(n²). It only finds the top eigenvector, so you skip the full O(n³) decomposition.

---

## Gotchas

**Q: Why should you not compare eigenvectors directly in a test?**
A: Sign and scale are ambiguous — `v` and `−v` are both valid eigenvectors. Check the defining property `A v ≈ λ v` instead.

**Q: Do `np.linalg.eig` eigenvalues come sorted?**
A: No — they are in no fixed order. Sort them before comparing.

**Q: Are eigenvalues always real?**
A: No. General matrices can have complex eigenvalues. Symmetric matrices always have real ones — use `eigh` for those.

**Q: Does the zero vector count as an eigenvector?**
A: No. `v = 0` trivially satisfies `A v = λ v`, so eigenvectors must be non-zero by definition.
