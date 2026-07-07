# SVD (Singular Value Decomposition) — Q&A

---

## Basic Concept

**Q: In one line, what does SVD say about a matrix?**
A: Any matrix is really three simple steps — rotate, then stretch, then rotate. SVD pulls those steps apart.

**Q: What are the three parts of `A = U Σ Vᵀ`?**
A: `Vᵀ` is a rotation of the input, `Σ` is a diagonal matrix that stretches along each axis, and `U` is a rotation of the output.

**Q: What does "orthogonal" mean for `U` and `V`?**
A: Their columns are perpendicular unit vectors. They only rotate/reflect — they never stretch. All stretching lives in `Σ`.

---

## Singular Values

**Q: What are singular values?**
A: The diagonal entries of `Σ`. Each one says how much the matrix stretches in one direction.

**Q: What two rules do singular values always follow?**
A: They are always ≥ 0, and they are sorted largest first: `σ₁ ≥ σ₂ ≥ … ≥ 0`.

**Q: What does a tiny singular value mean?**
A: A weak, barely-important direction — often just noise. Big singular value means an important direction with lots of stretch.

---

## Geometric Meaning

**Q: What happens if you apply a matrix to the unit circle?**
A: You get an ellipse. SVD explains it: `Vᵀ` rotates the circle, `Σ` stretches it into an ellipse, `U` rotates it to its final place.

**Q: What do the singular values equal, geometrically?**
A: The lengths of the ellipse's axes.

---

## Link to Eigenvalues

**Q: How are singular values related to eigenvalues?**
A: The singular values of `A` are the square roots of the eigenvalues of `AᵀA`: `σᵢ = √(λᵢ of AᵀA)`.

**Q: Why use `AᵀA` and not `A` itself?**
A: `AᵀA` is always symmetric with eigenvalues ≥ 0, so their square roots are real and non-negative — exactly what singular values must be.

**Q: What are the columns of `U` and `V` in terms of eigenvectors?**
A: Columns of `V` are eigenvectors of `AᵀA`; columns of `U` are eigenvectors of `AAᵀ`.

---

## Why AI Cares

**Q: How is SVD used for compression?**
A: Keep only the top-k singular values and drop the tiny ones. Rebuilding from those gives a much smaller matrix that is almost identical.

**Q: How does SVD relate to PCA?**
A: PCA is SVD applied to centered data. The top singular directions are the main patterns in the data.

**Q: What is the core reason SVD is so useful?**
A: Most of a matrix's "energy" sits in a few big singular values. Keep those, throw the rest away.

---

## Rank-k Approximation

**Q: What is a rank-k approximation?**
A: Sum only the top-k singular pieces: `Aₖ = Σ σᵢ uᵢ vᵢᵀ` for i = 1..k. Full k gives back `A` exactly; smaller k saves storage with small error.

**Q: How much storage does rank-k save?**
A: A full `m×n` matrix costs `m·n` numbers. Rank-k costs `k·(m + n + 1)`. For big matrices and small k, that's a huge saving.

---

## From Scratch

**Q: Why not compute full SVD by hand?**
A: It's hard. Instead, get the singular values from `AᵀA` (its eigenvalues, then square-rooted), and use a library for `U` and `V`.

**Q: How do you get eigenvalues of a 2×2 symmetric matrix by hand?**
A: With the quadratic formula: `λ = (trace ± √(trace² − 4·det)) / 2`, where trace and det come from the matrix.

**Q: What does the reconstruct function check?**
A: That `U @ diag(s) @ Vᵀ` multiplies back to the original `A` — proving the decomposition is correct.

---

## NumPy

**Q: What does `np.linalg.svd(A)` return?**
A: Three things: `U`, `s` (1-D array of singular values, sorted descending), and `Vt` (which is already `Vᵀ`).

**Q: Why can't you multiply `U @ s @ Vt` directly to rebuild `A`?**
A: `s` is a 1-D array, not a matrix. You must place the values on a diagonal first, then do `U @ diag(s) @ Vt`.

**Q: What is `full_matrices=False` for?**
A: A thin SVD for tall/wide matrices — smaller `U` and `Vt`, cheaper to compute.

---

## PyTorch

**Q: What does `torch.linalg.svd(A)` return?**
A: `U, S, Vh` — same layout as NumPy. `Vh` is `Vᵀ`, `S` is the 1-D singular values sorted descending.

**Q: Why prefer `torch.linalg.svd` over the old `torch.svd`?**
A: `torch.linalg.svd` returns `Vᵀ` to match NumPy. The old `torch.svd` returns `V`, which is easy to get wrong.

**Q: What dtype does torch SVD need?**
A: Float. Convert int tensors first.

---

## Complexity

**Q: How expensive is a full SVD?**
A: About O(min(m,n)·m·n) — expensive. For huge matrices, use truncated (top-k) SVD instead.

---

## Gotchas

**Q: Why shouldn't you compare `U` or `V` from two different SVD runs?**
A: Their columns have a sign-flip freedom — `u` and `−u` are both valid. Only the singular values are unique. Compare `s`, or check reconstruction.

**Q: Are singular values the same as eigenvalues?**
A: No. `σ = √(eigenvalue of AᵀA)`. Singular values are always ≥ 0; eigenvalues can be negative.

**Q: Common mistake with `Vt`?**
A: Transposing it again. NumPy/torch already return `Vᵀ`, so transposing gives back `V` and breaks the reconstruction.
