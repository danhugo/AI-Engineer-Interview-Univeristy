# Linear Regression: Normal Equation — Interview Knowledge Sheet

## Intuition

### Why "Linear"

Linear regression fits a straight line (2D), a plane (3D), or a hyperplane (many features) through the data points to model and forecast a relationship. The fit is always flat, never curved.

"Linear" means linear in the **weights**:

```
y_hat = w1*x1 + w2*x2 + b
```

The inputs `x` can be nonlinear (you can feed in `x²` or `x1*x2` as features). What matters is that the prediction is a linear combination of those features. The weights `w` are what make it "linear."

### Why "Regression"

"Regression" means to *return to a former state*. In the late 1800s, Francis Galton noticed that exceptionally tall or short parents had children with heights closer to the population average. He called this trend "regression toward the mean." The name stuck. Today it refers to any statistical model that predicts a continuous number.

### Why "Normal" Equation

"Normal" here is a geometric idea, not "normal" as in usual. It refers to a **normal vector** — a vector perpendicular to a surface.

The best-fit line (or plane) is the one where the total distance from points to the plane is smallest. At that best fit, the error vector `y - X·theta` is perpendicular (normal) to every column of `X`. That perpendicular condition is what produces the normal equation.

The normal equation is also the closed-form result of minimizing the squared error (Least Square). We take the derivative of the squared error, set it to zero, and solve. The solution is the normal equation.

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

We want `X·theta` as close as possible to `y`. But `X·theta` can only produce points in the **column space** of `X` — all weighted combinations of X's columns. Most of the time `y` is not in that space, so we cannot hit it exactly. We settle for the closest point in the space.

The closest point is the **orthogonal projection** of `y` onto the column space of `X` . At that best `theta`, the error vector `y - X·theta` is perpendicular to every column of `X`.

Writing "perpendicular to every column" as an equation:

```
Xᵀ(y - X·theta) = 0
```

Expand it:

```
XᵀX·theta - Xᵀy = 0
XᵀX·theta = Xᵀy
```

Solve for theta:

```
theta = (XᵀX)⁻¹Xᵀy
```

This is the normal equation. The name "normal" comes from that perpendicular condition.

### Same result from calculus

You can also reach the normal equation by minimizing squared error directly.

Start with the sum of squared errors in matrix form:

```
Error = (y - X·theta)ᵀ(y - X·theta)
```

To find the minimum, take the derivative with respect to `theta` and set it to zero. The result is the same `XᵀX·theta = Xᵀy`. So the geometry (perpendicular error) and the calculus (minimum squared error) agree.

### In code, do not compute the inverse

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
