# Linear Regression: Normal Equation — Interview Knowledge Sheet

## Intuition

**The normal-equation approach fits linear regression by directly solving the least-squares problem.**

Instead of training step by step, we ask:

```
Which weights make Xw closest to y?
```

In code, prefer a stable solver like `np.linalg.lstsq` or `torch.linalg.lstsq`.

---

## 1. The Model

Linear regression predicts a number:

```
y_hat = Xw + b
```

- `X` — input features
- `w` — learned feature weights
- `b` — bias/intercept
- `y_hat` — prediction

The bias lets the model shift predictions up or down even when all features are zero.

---

## 2. Bias Column Trick

Most linear algebra solvers expect one matrix multiplied by one parameter vector.

So we add a column of ones to `X`:

```
X = [[x1, x2]]

X_bias = [[1, x1, x2]]
theta = [b, w1, w2]
```

Then prediction becomes:

```
y_hat = X_bias @ theta
```

This keeps the bias and weights in one vector.

---

## 3. Normal Equation Intuition

The textbook formula is:

```
theta = (XᵀX)⁻¹Xᵀy
```

But in interviews and production code, the important point is:

**Do not explicitly compute the inverse.**

Use:

```
np.linalg.lstsq(X_bias, y, rcond=None)
torch.linalg.lstsq(X_bias, y)
```

These solve the same least-squares problem more safely.

---

## 4. Why Avoid the Matrix Inverse?

Computing an inverse can be:

- slower
- less numerically stable
- fragile when features are duplicated or nearly duplicated

Least-squares solvers are built for this exact job.

If an interviewer asks for the formula, you can give it. If they ask for code, use `lstsq`.

---

## 5. When This Method Is Useful

Use a direct least-squares solver when:

- the dataset fits in memory
- the number of features is not huge
- you want a simple exact baseline
- you are debugging a gradient-descent implementation

Use gradient descent instead when:

- the data is very large
- the model is not ordinary linear regression
- you want minibatches or deep learning tooling

---

## 6. NumPy Pattern

```python
Xb = add_bias_column(X)
theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
pred = Xb @ theta
```

`theta[0]` is the bias.

`theta[1:]` are the feature weights.

---

## 7. PyTorch Pattern

```python
Xb = add_bias_column_torch(X)
theta = torch.linalg.lstsq(Xb, y).solution
pred = Xb @ theta
```

Use tensors with floating-point dtype.

If `y` is shaped `(n,)`, the solution is shaped `(features + 1,)`.

If `y` is shaped `(n, 1)`, the solution is shaped `(features + 1, 1)`.

---

## Interview Gotchas

- Add the bias column before solving.
- Do not regularize here; this is ordinary least squares.
- Avoid `np.linalg.inv(X.T @ X)`.
- `lstsq` can still return a best-fit solution when the system is overdetermined.
- The loss being minimized is mean squared error, but the solver minimizes the equivalent sum of squared errors.
