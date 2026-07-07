# Matrix Transpose, Inverse & Multiply — Q&A

---

## Transpose

**Q: What does transpose do?**
A: It swaps rows and columns. Position `[i][j]` becomes `[j][i]`. A `(2,3)` matrix becomes `(3,2)`.

**Q: What is `(Aᵀ)ᵀ`?**
A: Just `A`. Transposing twice gets you back to the start.

**Q: What is `(AB)ᵀ`?**
A: `BᵀAᵀ` — the transpose of a product flips the order. It is NOT `AᵀBᵀ`.

**Q: Where does transpose show up in AI?**
A: Weight matrices — a batch linear layer is `X @ W.T`. Also `AᵀA` appears in least squares.

---

## Matrix Multiply

**Q: How does matrix multiply work?**
A: Dot every row of A with every column of B. Output `[i][j]` is row `i` of A dotted with column `j` of B.

**Q: What is the shape rule?**
A: `(m,n) · (n,p) = (m,p)`. The inner dimensions must match.

**Q: Is matrix multiply commutative?**
A: No. `AB ≠ BA` in general. Order matters.

---

## Determinant

**Q: How do you compute the determinant of a 2×2?**
A: `a*d - b*c` for `[[a,b],[c,d]]`. For `[[4,7],[2,6]]` it is `4*6 - 7*2 = 10`.

**Q: What does the determinant tell you?**
A: Whether the matrix can be inverted. If `det = 0`, there is no inverse (the matrix is singular).

**Q: What does the determinant mean geometrically?**
A: `|det|` is the factor by which the matrix scales area (2D) or volume (3D). A zero determinant means space was collapsed flat.

---

## Inverse

**Q: What is a matrix inverse?**
A: The "undo" of a matrix. `A @ A⁻¹ = I`, the identity matrix.

**Q: What is the identity matrix?**
A: The "1" of matrices — 1s on the diagonal, 0s elsewhere. Multiplying by it changes nothing.

**Q: What is the 2×2 inverse formula?**
A: Swap the diagonal, negate the off-diagonal, divide by the determinant: `A⁻¹ = (1/det) * [[d,-b],[-c,a]]`.

**Q: How do you check an inverse is correct?**
A: Multiply `A @ A⁻¹`. It should give the identity matrix (within rounding tolerance).

---

## Solving Linear Systems

**Q: How does the inverse solve `A x = b`?**
A: `x = A⁻¹ b`. Multiply both sides by `A⁻¹` on the left; `A⁻¹ A = I`, so `x = A⁻¹ b`.

**Q: What is the AI tie-in for transpose + multiply + inverse?**
A: Least squares / normal equations: `w = (Xᵀ X)⁻¹ Xᵀ y`. This is linear regression in closed form — it uses all three operations in one line.

---

## When Inverse Does Not Exist

**Q: When does a matrix have no inverse?**
A: When `det = 0`. The matrix is singular — its rows or columns are linearly dependent, so information is lost and it cannot be undone.

**Q: Can a non-square matrix have an inverse?**
A: No. Only square matrices can. Non-square matrices use the pseudo-inverse.

---

## From Scratch

**Q: How do you transpose a matrix from scratch?**
A: Build a new matrix where entry `[j][i]` comes from `[i][j]`: `[[A[i][j] for i in rows] for j in cols]`.

**Q: Why do `det2`/`inv2` only handle 2×2?**
A: General N×N determinant and inverse need Gaussian elimination — a lot more code. In real work you use a library.

---

## NumPy

**Q: How do you transpose, get determinant, and invert in NumPy?**
A: `A.T` for transpose, `np.linalg.det(A)` for determinant, `np.linalg.inv(A)` for inverse.

**Q: How do you make an identity matrix in NumPy?**
A: `np.eye(n)`.

**Q: What is the preferred way to solve `A x = b` in NumPy?**
A: `np.linalg.solve(A, b)` — faster and more stable than `inv(A) @ b`.

---

## PyTorch

**Q: How do you transpose, get determinant, and invert in PyTorch?**
A: `A.T`, `torch.linalg.det(A)`, `torch.linalg.inv(A)`.

**Q: How do you solve `A x = b` in PyTorch?**
A: `torch.linalg.solve(A, b)`. These functions need float tensors.

---

## Complexity

**Q: What is the cost of determinant and inverse for an N×N matrix?**
A: Both are O(n³). Transpose is O(m·n), or O(1) if returned as a view.

**Q: Why is transpose sometimes free?**
A: NumPy/PyTorch can return it as a view — they relabel the axes without copying data.

---

## Gotchas

**Q: Why prefer `solve(A, b)` over `inv(A) @ b`?**
A: `solve` is faster and more numerically stable. Computing a full inverse adds error and cost you do not need.

**Q: Why not compare inverse results with `==`?**
A: Inverses carry floating-point rounding error. Use a tolerance check like `allclose`.
