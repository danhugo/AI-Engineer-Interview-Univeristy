# Huber Loss — Interview Knowledge Sheet

## Intuition

Huber loss is a regression loss that mixes MSE and MAE.

For small errors:

```
behaves like MSE
```

For large errors:

```
behaves like MAE
```

This makes it smoother than MAE and less outlier-sensitive than MSE.

---

## 1. Formula

Let:

$$
e = \hat{y} - y
$$

With threshold `delta`:

$$
L_\delta(e) =
\begin{cases}
\frac{1}{2}e^2, & |e| \le \delta \\
\delta(|e| - \frac{1}{2}\delta), & |e| > \delta
\end{cases}
$$

---

## 2. Why It Helps

MSE gives large outliers huge influence.

MAE is more robust, but has a sharp corner at zero.

Huber keeps the smooth MSE shape near zero and switches to linear growth for big errors.

---

## 3. Delta

`delta` controls where the loss switches.

Small `delta`:

```
more MAE-like
```

Large `delta`:

```
more MSE-like
```

---

## 4. PyTorch Pattern

```python
loss = torch.nn.HuberLoss(delta=1.0)(pred, y)
```

PyTorch also has `SmoothL1Loss`, which is closely related but uses a slightly different scaling convention.

---

## 5. Interview Gotchas

- Huber is for regression.
- It is robust to outliers compared with MSE.
- It is smoother near zero than MAE.
- `delta` controls the switch from squared to linear behavior.
- Use Huber when outliers exist but you still want stable gradients.

---

## References

- PyTorch `HuberLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html
- PyTorch loss functions: https://docs.pytorch.org/docs/stable/nn.html#loss-functions
