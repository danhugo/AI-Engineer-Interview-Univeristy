# Lasso Regression with ISTA — Interview Knowledge Sheet

## Intuition

Lasso is **linear regression with an L1 penalty**.

The L1 penalty can push some weights exactly to zero. That means lasso can act like feature selection.

ISTA trains lasso with two moves:

```
gradient step -> soft-thresholding
```

The gradient step reduces prediction error.

The soft-thresholding step handles the L1 penalty.

---

## 1. The Lasso Objective

The model is still linear regression:

$$
\hat{y} = Xw + b
$$

The objective is:

$$
\text{loss}
= \frac{1}{n}\|Xw + b - y\|_2^2
+ \alpha\|w\|_1
$$

The L1 norm is:

$$
\|w\|_1 = \sum_{j=1}^{d}|w_j|
$$

Regularize `w`, not `b`.

---

## 2. Why L1 Creates Sparsity

L1 adds a constant pull toward zero.

Small weights can be pulled all the way to exactly zero.

That is different from ridge:

| Model | Penalty | Typical effect |
|-------|---------|----------------|
| Ridge | L2 | shrinks weights smoothly |
| Lasso | L1 | can set weights to zero |

A zero weight means the model ignores that feature.

---

## 3. Why ISTA?

The L1 penalty is not differentiable at zero.

So plain gradient descent is awkward.

ISTA uses a **proximal step** for the L1 part. In practice, that proximal step is soft-thresholding.

ISTA stands for:

```
Iterative Shrinkage-Thresholding Algorithm
```

---

## 4. Soft-Thresholding

Soft-thresholding shrinks values toward zero:

$$
S(x, t) = \operatorname{sign}(x)\max(|x|-t, 0)
$$

Examples with `t = 0.5`:

```
 2.0 ->  1.5
-2.0 -> -1.5
 0.2 ->  0.0
```

Values close enough to zero become exactly zero.

---

## 5. ISTA Update

First compute the MSE gradient:

$$
g_w = \frac{2}{n}X^T(Xw + b - y)
$$

$$
g_b = \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)
$$

Then update:

```
w = soft_threshold(w - lr * g_w, lr * alpha)
b = b - lr * g_b
```

Only `w` is soft-thresholded.

The bias gets a normal gradient update.

---

## 6. NumPy and PyTorch Pattern

Soft-thresholding in NumPy:

```python
def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)
```

Soft-thresholding in PyTorch:

```python
torch.sign(x) * torch.clamp(torch.abs(x) - t, min=0)
```

For this topic, manual updates are clearer than using a built-in optimizer.

---

## 7. Complexity

For `n` samples and `d` features, one ISTA step costs:

$$
O(nd)
$$

The expensive operations are `X @ w` and `X.T @ error`.

---

## 8. Interview Gotchas

- Lasso changes the loss, not the prediction formula.
- L1 can create exact zero weights.
- L1 is not differentiable at zero.
- ISTA uses soft-thresholding to handle the L1 part.
- Do not regularize the bias.
- Feature scaling matters because L1 penalizes coefficient size.
- Larger `alpha` usually means more zero weights.
