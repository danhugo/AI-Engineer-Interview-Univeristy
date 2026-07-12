# Lasso Regression with ISTA — Interview Knowledge Sheet

## Intuition

**Lasso is linear regression with an L1 penalty.** ISTA trains it by taking a normal gradient step, then applying soft-thresholding to the weights.

---

## 1. What Lasso Optimizes

Lasso uses the loss:

```
loss = mean((Xw + b - y)^2) + alpha * sum(abs(w))
```

The important interview detail:

- regularize `w`
- do not regularize `b`

The bias only shifts predictions up or down. Penalizing it usually makes the model worse for no useful feature-selection benefit.

---

## 2. Why L1 Creates Sparse Weights

L1 regularization adds a constant push toward zero.

That means small weights can become exactly zero.

Intuition:

- Ridge/L2 shrinks weights smoothly.
- Lasso/L1 can remove features completely.

This is why Lasso is often described as automatic feature selection.

---

## 3. Soft-Thresholding

Soft-thresholding is the key operation:

```
soft_threshold(x, t) = sign(x) * max(abs(x) - t, 0)
```

Examples with `t = 0.5`:

```
 2.0 ->  1.5
-2.0 -> -1.5
 0.2 ->  0.0
```

It shrinks every value toward zero. Values close enough to zero become exactly zero.

---

## 4. ISTA Training Loop

ISTA means **Iterative Shrinkage-Thresholding Algorithm**.

Each step does two things:

1. Take a gradient step on the MSE part.
2. Apply soft-thresholding for the L1 part.

```
pred = Xw + b
grad_w = (2 / n) * X.T @ (pred - y)
grad_b = (2 / n) * sum(pred - y)

w = soft_threshold(w - lr * grad_w, lr * alpha)
b = b - lr * grad_b
```

Only `w` gets soft-thresholded. The bias gets a normal gradient update.

---

## 5. NumPy and PyTorch View

In NumPy, implement ISTA directly with vectorized arrays.

In PyTorch, the same idea works with tensors:

```
torch.sign(x) * torch.clamp(torch.abs(x) - t, min=0)
```

For this interview-style exercise, manual tensor updates are clearer than building a full optimizer.

---

## 6. Interview Gotchas

- Lasso has no simple closed-form solution like ordinary least squares.
- L1 is not differentiable at zero, so proximal methods like ISTA are natural.
- Do not apply L1 to the bias term.
- Feature scaling matters because L1 penalizes coefficients directly.
- A larger `alpha` usually creates more zero weights.

---

## 7. Complexity

For `n` examples and `d` features, one ISTA step costs:

```
O(nd)
```

because the expensive work is `X @ w` and `X.T @ residual`.

