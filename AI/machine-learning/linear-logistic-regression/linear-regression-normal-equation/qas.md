# Linear Regression: Normal Equation — Q&A

---

## Core Idea

**Q: What does the normal-equation approach do?**
A: It directly solves for the linear-regression parameters that best fit the data in the least-squares sense.

**Q: What does linear regression predict?**
A: A continuous number, such as price, time, or score.

**Q: What is the prediction formula?**
A: `y_hat = Xw + b`.

---

## Bias Column

**Q: Why do we add a bias column?**
A: It lets us store the bias and weights in one parameter vector: `theta = [b, w1, w2, ...]`.

**Q: What value goes in the bias column?**
A: Ones.

**Q: After adding a bias column, how do we predict?**
A: `y_hat = X_bias @ theta`.

---

## Solver

**Q: What is the textbook normal equation?**
A: `theta = (XᵀX)⁻¹Xᵀy`.

**Q: Should you compute that inverse directly in code?**
A: No. Use `np.linalg.lstsq` or `torch.linalg.lstsq`.

**Q: Why avoid the inverse?**
A: It can be slower and less numerically stable, especially with repeated or highly correlated features.

**Q: What does `np.linalg.lstsq` return first?**
A: The fitted parameter vector.

---

## Usage

**Q: When is a direct least-squares solver a good choice?**
A: When the data fits in memory and the feature count is not huge.

**Q: When should you prefer gradient descent?**
A: For very large datasets, minibatch training, or models that do not have a simple closed-form solver.

**Q: What loss is ordinary least squares minimizing?**
A: Squared error, which corresponds to mean squared error up to a constant factor.

---

## NumPy and PyTorch

**Q: What is the NumPy pattern?**
A: Add a bias column, then call `np.linalg.lstsq(Xb, y, rcond=None)[0]`.

**Q: What is the PyTorch pattern?**
A: Add a bias column, then call `torch.linalg.lstsq(Xb, y).solution`.

**Q: Where is the bias stored in `theta`?**
A: `theta[0]`.

**Q: Where are the weights stored in `theta`?**
A: `theta[1:]`.
