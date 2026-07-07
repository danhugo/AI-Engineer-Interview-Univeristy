# Rank of a Matrix — Q&A

---

## What Rank Means

**Q: What is the rank of a matrix?**
A: The number of linearly independent rows — how many rows carry real, non-redundant information. It equals the number of independent columns too.

**Q: What does "linearly independent" mean?**
A: No row can be built from the others by scaling and adding. If a row is a copy or a multiple of another, it is dependent and does not add to the rank.

**Q: What is the rank of `[[1,2],[2,4]]`?**
A: 1. Row 2 is exactly 2 × row 1, so it is redundant. Only one row carries new information.

**Q: Is row rank the same as column rank?**
A: Yes, always. So you can count independent rows or independent columns — whichever is easier gives the same answer.

---

## Full Rank vs Rank-Deficient

**Q: What is the maximum possible rank of an m × n matrix?**
A: `min(m, n)`. You cannot have more independent rows than there are rows, or more independent columns than there are columns.

**Q: What is a full-rank matrix?**
A: One whose rank equals `min(m, n)` — every row (or column) is independent.

**Q: What is a rank-deficient matrix?**
A: One whose rank is less than `min(m, n)` — some rows are redundant.

**Q: For a square matrix, how do rank and invertibility relate?**
A: Full rank (rank = n) means invertible and determinant ≠ 0. Rank-deficient (rank < n) means singular, determinant = 0, no inverse.

---

## How to Find Rank

**Q: How do you find rank by row reduction?**
A: Use row operations to reduce the matrix to staircase form, then count the rows that are not all zero. Redundant rows collapse to zeros.

**Q: How do libraries compute rank?**
A: They count the non-zero singular values (from SVD). Values below a small tolerance are treated as zero. This is more stable with floating-point than raw elimination.

---

## Why AI Cares

**Q: What is collinearity and why does rank matter for it?**
A: Collinear features are copies or scaled versions of each other. They make the data matrix rank-deficient, which breaks linear regression — there is no unique solution.

**Q: What is low-rank structure used for?**
A: Big matrices often hold little real information. A low-rank approximation compresses them — used in image compression and recommendation systems.

**Q: How does LoRA relate to rank?**
A: LoRA fine-tunes a big model by adding a small low-rank update instead of changing all weights. Low rank means few real parameters, so it is cheap.

**Q: What happens to a rank-deficient system of equations?**
A: It has either no solution or infinitely many — never exactly one.

---

## From Scratch

**Q: How do you compute rank from scratch?**
A: Run Gaussian elimination on a float copy, count how many pivots you place. Each pivot is one independent row. Use a tolerance so tiny float noise counts as zero.

**Q: Why work on a copy in the pure-Python version?**
A: Elimination changes the matrix in place. Copying keeps the caller's input unchanged.

---

## NumPy

**Q: How do you get the rank in NumPy?**
A: `np.linalg.matrix_rank(M)`. It counts singular values above a tolerance and handles float noise for you.

---

## PyTorch

**Q: How do you get the rank in PyTorch?**
A: `torch.linalg.matrix_rank(M)`. Same idea as NumPy — it uses singular values. The tensor must be float.

---

## Complexity

**Q: What is the cost of computing rank?**
A: About O(m · n · min(m,n)) either way — roughly O(n³) for a square n × n matrix. The SVD method costs the same order but is numerically stable.

---

## Gotchas

**Q: Why must you reduce before counting rows?**
A: Redundant rows look independent until elimination collapses them to zero. Counting raw rows over-counts the rank.

**Q: Why use a tolerance instead of `== 0` on floats?**
A: Float math leaves tiny noise, so an exact zero check fails. Treat values below a small tolerance (like 1e-9) as zero.

**Q: Does adding more rows always raise the rank?**
A: No. A 100 × 2 matrix has rank at most 2. Rank is capped by `min(m, n)`.
