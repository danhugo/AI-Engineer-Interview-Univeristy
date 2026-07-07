# Matrix Decomposition (LU, QR, Cholesky) — Q&A

---

## Basic Concept

**Q: What is matrix decomposition?**
A: Factoring a matrix into simpler pieces (triangular, orthogonal), like `12 = 4 × 3`. The pieces make solving systems, least squares, and sampling fast and stable.

**Q: Why factor a matrix instead of computing its inverse?**
A: The inverse is slow, uses more memory, and loses accuracy. You factor once into triangular pieces, then solve with quick back-substitutions — and reuse the factors for many right-hand sides.

---

## LU Decomposition

**Q: What is LU decomposition?**
A: Splitting `A` into a lower triangular `L` and an upper triangular `U` so that `A = L U`. `L` has ones on its diagonal.

**Q: How do you get L and U?**
A: By Gaussian elimination. Each multiplier you use to zero out an entry below the diagonal is stored in `L`; the eliminated matrix becomes `U`.

**Q: What is pivoting and why is it needed?**
A: Swapping rows so the pivot is not 0 or tiny. Without it, LU can divide by zero or lose accuracy. With row swaps you get `P A = L U`, where `P` is a permutation matrix.

**Q: What is LU used for?**
A: Solving `A x = b` quickly, especially for many different `b`. It also gives the determinant as the product of `U`'s diagonal.

---

## QR Decomposition

**Q: What is QR decomposition?**
A: Splitting `A` into an orthogonal `Q` and an upper triangular `R`, so `A = Q R`.

**Q: What does "Q is orthogonal" mean?**
A: Its columns are unit vectors at right angles, so `Qᵀ Q = I`. That makes `Q` numerically stable — it never stretches or shrinks a vector.

**Q: Where does Q come from?**
A: From Gram-Schmidt on the columns of `A`, turning them into an orthonormal set.

**Q: What is QR used for?**
A: Least squares (fitting `A x ≈ b` with no exact answer) and as the engine inside eigenvalue algorithms. More stable than the normal equations `AᵀA`.

---

## Cholesky Decomposition

**Q: What is Cholesky decomposition?**
A: A faster LU for symmetric positive-definite matrices: `A = L Lᵀ`, where `L` is lower triangular.

**Q: What does symmetric positive-definite (SPD) mean?**
A: Symmetric (`A = Aᵀ`) and `xᵀ A x > 0` for every non-zero `x`. Covariance matrices are SPD.

**Q: Why prefer Cholesky over LU when you can?**
A: It is about twice as fast and very stable, and needs no pivoting.

**Q: What happens if you run Cholesky on a non-SPD matrix?**
A: It fails. That failure is a cheap way to test whether a matrix is SPD.

**Q: What is Cholesky used for in AI?**
A: Sampling from a multivariate Gaussian (`x = mean + L z`), covariance work, and optimization (Newton steps).

---

## From Scratch

**Q: How do you compute LU from scratch?**
A: Gaussian elimination: for each pivot column, compute the multiplier `U[i][k]/U[k][k]` for rows below, store it in `L`, and subtract to zero out entries below the pivot.

**Q: Why is from-scratch LU limited without pivoting?**
A: If a pivot is 0, dividing by it fails. It only works on matrices that happen not to need row swaps.

**Q: Why leave QR to the libraries?**
A: Gram-Schmidt must handle sign choices and near-zero columns carefully. It is easy to get subtly wrong, so libraries do it more reliably.

---

## NumPy

**Q: How do you do QR in NumPy?**
A: `Q, R = np.linalg.qr(A)`.

**Q: How do you do Cholesky in NumPy?**
A: `L = np.linalg.cholesky(A)` — it returns the lower `L` and raises `LinAlgError` if `A` is not SPD.

**Q: Does NumPy have plain LU?**
A: No. Use SciPy's `scipy.linalg.lu`, which returns the pivoted form `P A = L U`.

---

## PyTorch

**Q: How do you do QR and Cholesky in PyTorch?**
A: `Q, R = torch.linalg.qr(A)` and `L = torch.linalg.cholesky(A)`. Both need float tensors.

**Q: How do you do LU in PyTorch?**
A: `torch.linalg.lu_factor(A)` returns the LU data and pivots, mainly for solving systems.

---

## Complexity

**Q: What is the cost of these decompositions?**
A: All are `O(n³)` to factor an `n×n` matrix. Cholesky is about half the work of LU. Each solve after factoring is only `O(n²)`.

**Q: Why factor once instead of solving from scratch each time?**
A: Factoring is `O(n³)` once, but every later solve is only `O(n²)`. Reusing the factors for many right-hand sides is much cheaper.

---

## Gotchas

**Q: Why not compare L/Q/R directly between two libraries?**
A: Signs and ordering can differ, so the matrices may look different but be correct. Verify by reconstruction: `L@U≈A`, `Q@R≈A`, `L@Lᵀ≈A`.

**Q: Why prefer QR over normal equations for least squares?**
A: The normal equations `AᵀA` square the condition number, making the problem less stable. QR avoids that.
