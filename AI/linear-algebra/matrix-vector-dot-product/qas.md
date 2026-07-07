# Matrix, Vector & Dot Product — Q&A

---

## Vector Dot Product

**Q: What is a dot product?**
A: Take two vectors of the same length, multiply them element by element, then add all the products. The result is a single number (a scalar).

**Q: What is the formula for a dot product?**
A: `a · b = Σ aᵢ * bᵢ`. For `[1,2,3]·[4,5,6]` that is `1*4 + 2*5 + 3*6 = 32`.

**Q: Do the two vectors need the same length?**
A: Yes. Dot product pairs up elements one by one, so both vectors must have the same number of elements.

**Q: What does the dot product mean geometrically?**
A: `a · b = |a| * |b| * cos(θ)`. Positive means the vectors point the same way, zero means perpendicular (unrelated), negative means opposite directions.

**Q: Why does AI care about the dot product?**
A: It measures similarity. Cosine similarity, attention scores (`Q · K`), and a single neuron's output are all dot products.

---

## Matrix × Vector

**Q: How does matrix × vector work?**
A: Take one dot product per row of the matrix with the vector. Each row gives one number in the output.

**Q: What shape do you get from `(2,3) · (3,)`?**
A: `(2,)`. The matrix has 2 rows, so the answer has 2 numbers. The 3s match and cancel.

**Q: Where does matrix × vector show up in AI?**
A: It is exactly what one neural network layer does to a single input: `W x`.

---

## Matrix × Matrix

**Q: How do you compute matrix × matrix?**
A: Dot every row of A with every column of B. Output position `[i][j]` is row `i` of A dotted with column `j` of B.

**Q: For `A B`, what goes in position `[0][1]`?**
A: Row 0 of A dotted with column 1 of B.

---

## Shape Rules

**Q: What is the shape rule for multiplying two matrices?**
A: `(m,n) · (n,p) = (m,p)`. The inner dimensions (the two touching numbers) must be equal; the outer ones become the result shape.

**Q: Why does `(2,3) · (2,4)` fail?**
A: The inner dimensions are 3 and 2. They are not equal, so the shapes do not line up and it errors.

**Q: Is matrix multiplication commutative?**
A: No. `A B ≠ B A` in general. Order matters.

**Q: Quick trick to check if a matmul is valid?**
A: Write the two shapes side by side. If the two touching numbers are equal, it works and they cancel out.

---

## Neural Networks

**Q: What is a linear (dense) layer in math form?**
A: `y = W x + b`, where `x` is the input, `W` is the learned weight matrix, `b` is the learned bias, and `y` is the output.

**Q: How is a batch of inputs handled?**
A: Use matrix × matrix: `Y = X Wᵀ + b`, where each row of `X` is one sample. Output shape is `(batch, out_features)`.

**Q: What is a deep network in terms of linear layers?**
A: Many linear layers (`W x + b`) stacked, with a non-linear function between them.

---

## From Scratch in Pure Python

**Q: How do you write a dot product by hand?**
A: Start a total at 0, loop over the indices, add `a[i] * b[i]` each step, return the total.

**Q: How does from-scratch matrix×vector build on the dot product?**
A: It is one dot product per row: `[dot(row, x) for row in W]`.

**Q: How does from-scratch matrix×matrix build on the dot product?**
A: It is a dot product for every row×column pair. For each row of A and each column j of B, compute `dot(row, column_j)`.

**Q: Should you use these hand-written loops in real work?**
A: No — they are slow in Python. They are for learning. NumPy and PyTorch do the same math with fast, tuned kernels.

---

## NumPy

**Q: How do you compute a dot product in NumPy?**
A: `np.dot(a, b)` or `a @ b`. The `@` operator is the cleanest.

**Q: What does `@` do in NumPy?**
A: It is the matmul operator. It handles vector·vector, matrix·vector, and matrix·matrix depending on the shapes.

**Q: What is the difference between `A * B` and `A @ B`?**
A: `*` is elementwise multiply (multiply matching positions). `@` is real matrix multiplication. Using `*` when you meant matmul is a common bug.

**Q: What does `.T` do?**
A: Transpose — it flips rows and columns. A `(2,3)` matrix becomes `(3,2)`.

**Q: What is broadcasting used for here?**
A: It lets you add a bias vector to every row automatically, like `X @ W.T + b`.

---

## PyTorch

**Q: How do you compute a dot product in PyTorch?**
A: `torch.dot(a, b)` for two 1-D tensors. It returns a scalar tensor.

**Q: How do you do matrix × vector in PyTorch?**
A: `torch.mv(W, a)` — mv stands for matrix-vector.

**Q: How do you do matrix × matrix in PyTorch?**
A: `torch.matmul(A, B)` or `A @ B`.

**Q: What is the difference between `torch.matmul` and `torch.dot`/`torch.mv`/`torch.bmm`?**
A: `torch.matmul` and `@` are general — they pick dot / mv / matmul / batched automatically from the shapes. `torch.dot`, `torch.mv`, `torch.bmm` are strict versions for one specific shape.

**Q: What is `torch.bmm` for?**
A: Batch matrix multiply — it multiplies each matrix in a batch of matrices (3-D tensors).

**Q: A common dtype trap in PyTorch?**
A: `torch.dot` needs both tensors to have the same dtype. Mixing int and float errors, so use floats.

---

## Complexity

**Q: What is the time cost of a dot product, matrix×vector, and matrix×matrix?**
A: Dot product `(n)` is O(n). Matrix×vector `(m,n)` is O(m·n). Matrix×matrix `(m,n)·(n,p)` is O(m·n·p).

**Q: Should you write your own matmul loop in Python?**
A: No. NumPy and PyTorch use tuned BLAS / GPU kernels that are far faster. Always use the library.

---

## Gotchas

**Q: You get a shape mismatch error. First thing to do?**
A: Print the shapes of both operands. Check that the inner dimensions match (`(m,n)·(n,p)`).

**Q: What is cosine similarity vs a plain dot product?**
A: Cosine similarity is the dot product of two **normalized** (unit-length) vectors. It measures only direction, not magnitude.
