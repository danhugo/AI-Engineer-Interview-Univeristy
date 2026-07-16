# Elastic Net Gradient Descent — Interview Knowledge Sheet

## Intuition

Elastic Net is **linear regression with both L1 and L2 penalties**.

It combines:

- lasso's feature selection
- ridge's smooth shrinkage

The model is still:

```
y_hat = Xw + b
```

Only the loss changes.

---

## 1. The Objective

Elastic Net uses:

$$
\text{loss}
= \frac{1}{n}\|Xw + b - y\|_2^2
+ \lambda_1\|w\|_1
+ \lambda_2\|w\|_2^2
$$

Where:

- `lambda_1` controls the L1 penalty
- `lambda_2` controls the L2 penalty

Regularize `w`, not `b`.

---

## 2. Why Combine L1 and L2?

L1 can set weights exactly to zero:

```
feature selection
```

L2 discourages large weights smoothly:

```
stability
```

Elastic Net is useful when features are correlated. Pure lasso may choose one feature from a correlated group and ignore the others. Elastic Net can keep groups more stable while still shrinking weak features.

---

## 3. Relation to Ridge and Lasso

Elastic Net includes ridge and lasso as special cases.

| Penalties | Model |
|-----------|-------|
| `lambda_1 > 0`, `lambda_2 = 0` | lasso |
| `lambda_1 = 0`, `lambda_2 > 0` | ridge |
| `lambda_1 > 0`, `lambda_2 > 0` | elastic net |

This makes Elastic Net a middle ground.

---

## 4. Gradients and Subgradients

The MSE weight gradient is:

$$
g_{\text{mse},w}
= \frac{2}{n}X^T(Xw + b - y)
$$

The bias gradient is:

$$
g_b
= \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
$$

The L2 gradient is:

$$
2\lambda_2 w
$$

The L1 term uses a subgradient:

$$
\lambda_1\operatorname{sign}(w)
$$

At zero, the L1 subgradient is a range. Simple interview code often uses `sign(0) = 0`.

---

## 5. Gradient Descent Update

Using a simple subgradient update:

```
pred = X @ w + b
error = pred - y

grad_w = (2 / n) * X.T @ error
grad_w += lambda_1 * sign(w)
grad_w += 2 * lambda_2 * w

grad_b = (2 / n) * sum(error)

w -= lr * grad_w
b -= lr * grad_b
```

Only `w` gets regularization terms.

The bias uses only the prediction-error gradient.

---

## 6. Subgradient vs Proximal Update

The simple update above is easy to explain.

But L1 penalties are often handled better with proximal methods, like soft-thresholding.

Why?

- subgradient descent may not create exact zeros cleanly
- proximal updates are designed for non-smooth penalties

For interview code, the simple subgradient version is acceptable if you explain the tradeoff.

---

## 7. Complexity

For `n` samples and `d` features, one gradient step costs:

$$
O(nd)
$$

The expensive operations are `X @ w` and `X.T @ error`.

---

## 8. Interview Gotchas

- Elastic Net combines L1 and L2 penalties.
- L1 can create sparsity.
- L2 improves stability when features are correlated.
- Do not regularize the bias.
- Feature scaling matters.
- The L1 term is not differentiable at zero.
- Simple subgradient descent is easy to explain, but proximal methods handle L1 better.
