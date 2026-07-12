# Elastic Net Gradient Descent — Interview Knowledge Sheet

## Intuition

**Elastic Net is linear regression with both L1 and L2 penalties.** It combines Lasso-style feature selection with Ridge-style stability.

---

## 1. Objective

Elastic Net uses:

```
loss = mean((Xw + b - y)^2)
     + l1 * sum(abs(w))
     + l2 * sum(w^2)
```

As with most regularized linear models:

- regularize the weights `w`
- do not regularize the bias `b`

The bias is not a feature strength. It is just the baseline prediction.

---

## 2. Why Combine L1 and L2?

L1 helps with sparsity:

```
some weights become exactly zero
```

L2 helps with stability:

```
large weights are discouraged smoothly
```

Elastic Net is useful when features are correlated. Pure Lasso may choose one correlated feature and ignore the others. Elastic Net often spreads weight more smoothly while still shrinking unhelpful features.

---

## 3. Gradients and Subgradients

The MSE part has the normal gradient:

```
pred = Xw + b
residual = pred - y

grad_mse_w = (2 / n) * X.T @ residual
grad_mse_b = (2 / n) * sum(residual)
```

The L2 penalty gradient is:

```
2 * l2 * w
```

The L1 penalty uses a subgradient:

```
l1 * sign(w)
```

At exactly zero, the true subgradient is a range. In simple interview code, `sign(0) = 0` is a common practical choice.

---

## 4. Gradient Descent Update

The weight gradient is:

```
grad_w = grad_mse_w + l1 * sign(w) + 2 * l2 * w
```

The bias gradient is only:

```
grad_b = grad_mse_b
```

Then update:

```
w = w - lr * grad_w
b = b - lr * grad_b
```

---

## 5. Interview Gotchas

- Do not regularize the bias.
- L1 uses a subgradient, not an ordinary derivative everywhere.
- L2 gradient is `2 * l2 * w` when the penalty is `l2 * sum(w^2)`.
- Feature scaling matters because regularization penalizes coefficient size.
- Gradient descent needs a reasonable learning rate.

---

## 6. Complexity

One gradient descent step costs:

```
O(nd)
```

for `n` examples and `d` features.

