# Polynomial Regression Fit — Interview Knowledge Sheet

## Intuition

**Polynomial regression fits curves by expanding the input into powers, then running ordinary linear regression.**

---

## 1. The Trick

For one input feature `x`, create new features:

```
x -> [1, x, x^2, x^3]
```

Then fit:

```
y_hat = theta0 + theta1*x + theta2*x^2 + theta3*x^3
```

The prediction is curved in `x`, but the model is still linear in the parameters `theta`.

That is why polynomial regression is still a linear model.

---

## 2. Polynomial Feature Matrix

For `x = [0, 1, 2]` and degree `2`, the feature matrix is:

```
[[1, 0, 0],
 [1, 1, 1],
 [1, 2, 4]]
```

Each row is:

```
[1, x, x^2]
```

The first column of ones is the bias/intercept feature.

---

## 3. Fitting with Least Squares

After building the feature matrix `Phi`, fit with least squares:

```
theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
```

In PyTorch, use:

```
torch.linalg.lstsq(Phi, y).solution
```

Do not explicitly compute the matrix inverse in interview code. Least-squares solvers are more stable.

---

## 4. Prediction

Prediction repeats the same feature expansion:

```
Phi_new = polynomial_features(x_new, degree)
y_pred = Phi_new @ theta
```

The most common bug is fitting with one feature order and predicting with a different feature order.

---

## 5. Overfitting

High-degree polynomials can bend too much.

Signs of overfitting:

- training error is very low
- validation/test error is high
- the curve wiggles wildly between points

Keep the degree small unless you have enough data and validation evidence.

---

## 6. Scaling

Powers can grow quickly:

```
100^5 = 10,000,000,000
```

Large powers can make fitting numerically unstable. In real projects, scale `x` before creating polynomial features.

---

## 7. Complexity

Creating polynomial features for `n` examples and degree `d` costs:

```
O(nd)
```

Least-squares fitting can be more expensive, depending on the solver and degree.

