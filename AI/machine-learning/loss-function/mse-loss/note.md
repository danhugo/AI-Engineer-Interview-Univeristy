# MSE Loss — Interview Knowledge Sheet

## Intuition

Mean squared error measures how far predicted numbers are from true numbers.

It squares each error:

```
small error -> small penalty
large error -> much larger penalty
```

So MSE cares a lot about large mistakes.

---

## 1. Formula

For predictions `y_hat` and targets `y`:

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
$$

The best possible value is `0`.

---

## 2. Why Square the Error?

Squaring does three things:

- removes the sign of the error
- punishes large errors strongly
- gives a smooth gradient

Example:

$$
10^2 = 100
$$

One large outlier can dominate MSE.

---

## 3. Gradient

For one prediction:

$$
\frac{\partial}{\partial \hat{y}}(\hat{y}-y)^2 = 2(\hat{y}-y)
$$

For the mean over `n` samples:

$$
\nabla_{\hat{y}}\text{MSE} = \frac{2}{n}(\hat{y}-y)
$$

This is why linear regression gradients often contain `2 / n`.

---

## 4. When to Use

Use MSE when:

- the target is continuous
- large errors should be punished strongly
- outliers are meaningful, not just noise

Avoid MSE when outliers should not dominate. Use MAE or Huber instead.

---

## 5. PyTorch Pattern

```python
loss = torch.nn.MSELoss()(pred, y)
```

By default, PyTorch returns the mean loss.

---

## 6. Interview Gotchas

- MSE is for regression, not classification.
- Squaring makes MSE sensitive to outliers.
- RMSE is often easier to interpret because it has target units.
- The gradient points in the direction of prediction error.
- Check whether your implementation uses mean or sum.

---

## References

- scikit-learn `mean_squared_error`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
- PyTorch `MSELoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
