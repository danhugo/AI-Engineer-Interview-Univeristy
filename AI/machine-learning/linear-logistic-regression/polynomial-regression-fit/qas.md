# Polynomial Regression Fit — Q&A

---

## Core Idea

**Q: What is polynomial regression?**
A: Linear regression on expanded features like `[1, x, x^2, x^3]`.

**Q: Why is polynomial regression still considered linear regression?**
A: It is linear in the learned parameters, even though the curve is nonlinear in `x`.

**Q: What does degree mean?**
A: The highest power of `x` included in the feature matrix.

---

## Feature Matrix

**Q: What features do we create for degree 2?**
A: `[1, x, x^2]`.

**Q: Why include a column of ones?**
A: It lets the model learn an intercept/bias term.

**Q: What is a common implementation bug?**
A: Using a different feature order during prediction than during fitting.

---

## Fitting

**Q: How do you fit polynomial regression in NumPy?**
A: Build the polynomial feature matrix, then call `np.linalg.lstsq`.

**Q: Why not compute the inverse directly?**
A: Least-squares solvers are usually more numerically stable.

**Q: What is the PyTorch equivalent?**
A: `torch.linalg.lstsq(Phi, y).solution`.

---

## Prediction

**Q: How do you predict on new x values?**
A: Build polynomial features for the new values and multiply by `theta`.

**Q: What shape should theta have for degree 3?**
A: Four values: bias, x weight, x² weight, and x³ weight.

---

## Overfitting and Scaling

**Q: Why can high-degree polynomials overfit?**
A: They can create very flexible curves that memorize training points.

**Q: What does polynomial overfitting often look like?**
A: A curve that wiggles wildly between data points.

**Q: Why does scaling matter?**
A: Powers of large inputs can become huge and make fitting unstable.

**Q: What is a practical interview recommendation?**
A: Start with a low degree, scale features, and choose degree using validation data.

