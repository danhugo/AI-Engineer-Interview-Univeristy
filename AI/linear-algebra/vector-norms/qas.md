# Vector & Matrix Norms — Q&A

---

## Basic Concept

**Q: What is a norm?**
A: The size (length) of a vector — one number saying how big it is. Different norms measure "big" in different ways.

**Q: If someone says "the norm" with no other word, which one do they mean?**
A: The L2 (Euclidean) norm — the ordinary straight-line length.

---

## L2 Norm

**Q: How do you compute the L2 norm?**
A: Square each element, add them up, take the square root: `√(Σ vᵢ²)`. For `[3, -4]` it is `√25 = 5`.

**Q: Where is L2 used in AI?**
A: Weight decay (L2 regularization), distance between embeddings, and gradient clipping by norm.

---

## L1 Norm

**Q: How do you compute the L1 norm?**
A: Add up the absolute values: `Σ |vᵢ|`. For `[3, -4]` it is `3 + 4 = 7`.

**Q: What does L1 regularization do to weights?**
A: It pushes weights to exactly zero, making the model sparse (many zeros). Useful for feature selection.

---

## L-infinity Norm

**Q: How do you compute the L-infinity norm?**
A: Take the largest absolute value: `max |vᵢ|`. For `[3, -4]` it is `4`.

**Q: Where is L-infinity used?**
A: Gradient clipping by max value, and some robustness bounds.

---

## General Lp Norm

**Q: What is the general Lp norm formula?**
A: `||v||ₚ = (Σ |vᵢ|ᵖ)^(1/p)`. p=1 gives L1, p=2 gives L2, p→∞ gives L-infinity.

**Q: What does a bigger p do?**
A: It puts more weight on the largest element. At p→∞, only the max matters.

---

## Frobenius Norm

**Q: What is the Frobenius norm?**
A: The L2 norm applied to all entries of a matrix as one long vector: square every entry, add, take the square root. `||M||_F = √(Σ Mᵢⱼ²)`.

**Q: Why is the Frobenius norm useful in AI?**
A: It measures the overall size of a weight matrix — used in regularization and to see how much weights changed during training.

---

## Normalization

**Q: How do you normalize a vector?**
A: Divide it by its L2 norm: `v̂ = v / ||v||₂`. The result points the same way but has length 1 (a unit vector).

**Q: Why normalize vectors in AI?**
A: Normalized vectors make cosine similarity just a dot product. Embeddings are often normalized so only direction matters, not magnitude.

**Q: What breaks when you normalize a zero vector?**
A: The L2 norm is 0, so dividing gives NaN/inf. Guard against it — skip it or add a tiny epsilon.

---

## From Scratch

**Q: How do you write L2 norm by hand in Python?**
A: `sum(x * x for x in v) ** 0.5` — sum the squares, then `** 0.5` is the square root.

**Q: How is Frobenius norm coded from scratch?**
A: Same as L2 but loop over every entry of the matrix: `sum(x*x for row in M for x in row) ** 0.5`.

---

## NumPy

**Q: How do you compute norms in NumPy?**
A: `np.linalg.norm(v)`. It defaults to L2. Use `ord=1` for L1, `ord=np.inf` for L-infinity.

**Q: How do you get the Frobenius norm of a matrix in NumPy?**
A: `np.linalg.norm(M)` (default for a 2-D array is Frobenius) or `np.linalg.norm(M, ord='fro')`.

**Q: What is the trap with `ord=2` on a 2-D array?**
A: For a matrix, `ord=2` means the matrix 2-norm (largest singular value), not Frobenius. Use `ord='fro'` for Frobenius.

**Q: How do you get the norm of each row?**
A: Pass `axis=1`: `np.linalg.norm(M, axis=1)`. Use `axis=0` for columns.

---

## PyTorch

**Q: How do you compute norms in PyTorch?**
A: `torch.linalg.norm(v)` — L2 by default. Use `ord=1`, `ord=float('inf')`, or `ord='fro'` for others.

**Q: What is the easy way to normalize in PyTorch?**
A: `torch.nn.functional.normalize(v, dim=0)` — unit vector, L2 by default.

**Q: What dtype do torch norms need?**
A: Float. Passing an int tensor errors — use float tensors.

---

## Complexity

**Q: What is the cost of computing a norm?**
A: O(n) for vector norms (one pass), O(m·n) for Frobenius (every matrix entry once). All are a single cheap pass.

---

## Gotchas

**Q: You wrote `Σ vᵢ²` and called it the L2 norm. What is wrong?**
A: That is the *squared* norm. The real L2 norm needs the square root on top.

**Q: L1 vs L2 regularization — what is the difference in effect?**
A: L1 makes weights exactly zero (sparse). L2 makes weights small but usually non-zero.
