# Polynomial Regression Fit — Interview Knowledge Sheet

## Intuition

Polynomial regression is **linear regression after feature expansion**.

Linear regression can only draw a straight line if the input is just `x`. Polynomial regression gives the model more inputs:

```
x -> [1, x, x^2, x^3]
```

Then ordinary linear regression fits weights on those expanded features.

The model:

$$
\hat{y} = \theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3
$$

The curve is nonlinear in `x`. But the model is still linear in the parameters `theta`.

That is the main trick.

---

## 1. Why It Is Still Linear Regression

A model is called linear when it is linear in its learned weights.

This is linear in the weights:

$$
\hat{y} = \theta_0 + \theta_1x + \theta_2x^2
$$

The learned values are `theta0`, `theta1`, and `theta2`. None of them are squared or multiplied together.

So polynomial regression is not a new optimizer. It is ordinary least squares on a new feature matrix.

---

## 2. Polynomial Features

For one feature and degree `3`, create:

$$
\phi(x) = [1, x, x^2, x^3]
$$

For `x = [0, 1, 2]` and degree `2`, the feature matrix is:

```
[[1, 0, 0],
 [1, 1, 1],
 [1, 2, 4]]
```

Each row is:

$$
[1, x, x^2]
$$

The first column of ones is the intercept feature.

### Multiple input features

With two inputs `[a, b]` and degree `2`, polynomial features can include interaction terms:

$$
[1, a, b, a^2, ab, b^2]
$$

The `ab` term lets the model learn that two features matter together.

---

## 3. Fitting With Least Squares

After building the polynomial feature matrix `Phi`, fit the same way as linear regression:

$$
\theta = \arg\min_\theta \|\Phi\theta - y\|_2^2
$$

In NumPy:

```
theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
```

In PyTorch:

```
theta = torch.linalg.lstsq(Phi, y).solution
```

Do not explicitly compute the inverse in interview code. Least-squares solvers are more stable.

---

## 4. Prediction

Prediction must use the same feature expansion as training:

```
Phi_new = polynomial_features(x_new, degree)
y_pred = Phi_new @ theta
```

The most common bug is changing feature order.

If training uses:

$$
[1, x, x^2]
$$

prediction must use the same order.

---

## 5. Degree Controls Flexibility

The degree controls how much the curve can bend.

| Degree | Shape |
|--------|-------|
| 1 | straight line |
| 2 | one bend |
| 3 | S-like curves |
| high | can wiggle a lot |

High degree can fit the training data very well. It can also overfit.

Signs of overfitting:

- training error is very low
- validation or test error is high
- the curve wiggles between points

Use validation data to choose the degree.

---

## 6. Scaling

Powers can grow quickly:

$$
100^5 = 10{,}000{,}000{,}000
$$

Large values make the feature matrix poorly scaled. That can make fitting unstable.

In real projects, scale `x` before creating polynomial features.

---

## 7. Complexity

For one input feature, creating degree `d` features for `n` samples costs:

$$
O(nd)
$$

With many input features, the number of polynomial features grows quickly. Degree `2` with `[a, b]` is small. Degree `5` with many columns can become huge.

Least-squares fitting is usually the expensive part after feature expansion.

---

## 8. Interview Gotchas

- Polynomial regression is linear regression on expanded features.
- The curve is nonlinear in `x`, but linear in the weights.
- Always use the same feature order for training and prediction.
- Include or handle the bias term consistently.
- Scale features before taking high powers.
- High degree can overfit badly.
- Use `np.linalg.lstsq`, not a manual matrix inverse.
