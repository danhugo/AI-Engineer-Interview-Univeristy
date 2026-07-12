# Ridge Regression Loss — Interview Knowledge Sheet

## Intuition

**Ridge regression is linear regression plus a penalty for large weights.**

It uses mean squared error for prediction quality and an L2 penalty for simpler weights:

```
loss = MSE + alpha * sum(w²)
```

---

## 1. Why Regularize?

Linear regression can overfit when:

- there are many features
- features are correlated
- the dataset is small or noisy

Ridge discourages very large weights, which often makes the model more stable on new data.

---

## 2. The Model

The prediction is still ordinary linear regression:

```
y_hat = Xw + b
```

Ridge does not change the prediction formula.

It changes the loss used to choose the parameters.

---

## 3. The Ridge Loss

The loss has two parts:

```
prediction_loss = mean((y_hat - y)²)
penalty = alpha * sum(w²)
total_loss = prediction_loss + penalty
```

`alpha` controls the strength of regularization.

- `alpha = 0` means ordinary linear regression
- larger `alpha` means stronger weight shrinking

---

## 4. Do Not Regularize the Bias

The bias is usually not regularized.

Regularizing the bias would punish the model for shifting predictions up or down. That usually does not help with overfitting in the same way large feature weights do.

So if:

```
theta = [b, w1, w2, w3]
```

the penalty should use only:

```
theta[1:]
```

---

## 5. NumPy Pattern

```python
pred = X @ w + b
mse = np.mean((pred - y) ** 2)
penalty = alpha * np.sum(w ** 2)
loss = mse + penalty
```

Keep the function small and explicit.

---

## 6. PyTorch Pattern

```python
pred = model(X)
mse = torch.nn.MSELoss()(pred, y)
penalty = alpha * torch.sum(model.weight ** 2)
loss = mse + penalty
```

For `torch.nn.Linear`, the bias is stored separately in `model.bias`, so only regularize `model.weight`.

---

## 7. Ridge vs Lasso

Ridge uses an L2 penalty:

```
sum(w²)
```

It shrinks weights smoothly but usually does not make them exactly zero.

Lasso uses an L1 penalty:

```
sum(abs(w))
```

It can push some weights exactly to zero.

---

## Interview Gotchas

- Ridge changes the loss, not the prediction formula.
- Do not regularize the bias term.
- `alpha` is a hyperparameter.
- Bigger `alpha` usually means smaller weights but potentially more underfitting.
- In PyTorch, add the penalty to the loss before `backward()`.
