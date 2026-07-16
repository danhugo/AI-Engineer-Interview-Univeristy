# Ridge Regression Loss — Interview Knowledge Sheet

## Intuition

Ridge regression is **linear regression with a penalty for large weights**.

Ordinary linear regression only asks:

```
How close are the predictions to y?
```

Ridge also asks:

```
Are the weights too large?
```

The model is still:

```
y_hat = Xw + b
```

Only the loss changes.

---

## 1. Why Regularize?

Linear regression can become unstable when:

- features are highly correlated
- there are many features
- the dataset is small or noisy

In those cases, many different weight vectors can fit the training data almost equally well. Some of those vectors use very large weights.

Large weights make predictions sensitive to small input changes.

Ridge discourages that by preferring smaller weights.

---

## 2. The Ridge Objective

The ridge loss is:

$$
\text{loss}
= \frac{1}{n}\|Xw + b - y\|_2^2
+ \alpha\|w\|_2^2
$$

The first term measures prediction error.

The second term penalizes large weights:

$$
\|w\|_2^2 = \sum_{j=1}^{d} w_j^2
$$

`alpha` controls regularization strength.

| Alpha | Effect |
|-------|--------|
| `0` | ordinary linear regression |
| small | light shrinkage |
| large | strong shrinkage, possible underfitting |

---

## 3. What L2 Does

The L2 penalty grows quickly for large weights.

Example:

$$
10^2 = 100
$$

So ridge strongly discourages one feature from getting a huge coefficient.

It usually shrinks weights toward zero, but does not make them exactly zero.

That is the key difference from lasso.

---

## 4. Bias Term

Usually do not regularize the bias.

If:

```
theta = [b, w1, w2, w3]
```

the ridge penalty should use only:

```
theta[1:]
```

The bias is the baseline prediction. Penalizing it does not control feature complexity in the same way.

---

## 5. Gradient

For:

$$
\text{loss}
= \frac{1}{n}\|Xw + b - y\|_2^2
+ \alpha\|w\|_2^2
$$

the gradients are:

$$
\nabla_w =
\frac{2}{n}X^T(Xw + b - y) + 2\alpha w
$$

$$
\nabla_b =
\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)
$$

The bias gradient has no ridge term.

---

## 6. NumPy Pattern

```python
pred = X @ w + b
error = pred - y

mse = np.mean(error ** 2)
penalty = alpha * np.sum(w ** 2)
loss = mse + penalty

dw = (2 / n) * X.T @ error + 2 * alpha * w
db = (2 / n) * np.sum(error)
```

---

## 7. PyTorch Pattern

```python
pred = model(X)
mse = torch.nn.MSELoss()(pred, y)
penalty = alpha * torch.sum(model.weight ** 2)
loss = mse + penalty

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

For `torch.nn.Linear`, regularize `model.weight`, not `model.bias`.

---

## 8. Ridge vs Lasso

| Model | Penalty | Effect |
|-------|---------|--------|
| Ridge | L2, `sum(w^2)` | smooth shrinkage |
| Lasso | L1, `sum(abs(w))` | can create exact zeros |

Use ridge when you want stability and do not need feature selection.

---

## 9. Interview Gotchas

- Ridge changes the loss, not the prediction formula.
- `alpha` is a hyperparameter.
- Larger `alpha` means stronger shrinkage.
- Ridge helps with correlated features and unstable weights.
- Ridge usually does not create exact zero weights.
- Do not regularize the bias.
- Feature scaling matters because the penalty acts on coefficient size.
