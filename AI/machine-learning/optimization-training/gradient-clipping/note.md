# Gradient Clipping — Interview Knowledge Sheet

## Intuition

Gradient clipping limits an update when the gradient is too large.

It does not change the loss function.

It changes the gradient before the optimizer step:

```python
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

This is most common when gradients can explode, especially in recurrent networks and deep sequence models.

---

## 1. Exploding Gradients

An exploding gradient is a gradient with very large magnitude.

If the optimizer uses it directly, one step can throw parameters far away:

$$
\theta \leftarrow \theta - \eta g
$$

If `g` is huge, even a modest learning rate can produce a huge update.

---

## 2. Clip by Norm

Norm clipping rescales the whole gradient vector only when its norm is too large.

Let:

$$
\|g\|_2 = \sqrt{\sum_i g_i^2}
$$

If:

$$
\|g\|_2 > c
$$

then:

$$
g \leftarrow g \frac{c}{\|g\|_2}
$$

This keeps the gradient direction but limits its size.

---

## 3. Clip by Value

Value clipping clamps each gradient element independently:

$$
g_i \leftarrow \min(\max(g_i, -c), c)
$$

This is simple, but it can change the gradient direction more than norm clipping.

---

## 4. PyTorch Pattern

Use norm clipping after `backward()` and before `step()`:

```python
optimizer.zero_grad()
loss = loss_fn(model(x), y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

`clip_grad_norm_` modifies gradients in place and returns the total norm before clipping.

---

## 5. What Clipping Does Not Fix

Gradient clipping is a stabilizer, not a cure for every training problem.

It does not replace:

- proper learning-rate tuning
- sensible initialization
- normalization when needed
- checking for numerical bugs

If clipping activates on almost every step, the learning rate may be too high or the model may be unstable.

---

## 6. Interview Gotchas

- Clip gradients after `loss.backward()`, before `optimizer.step()`.
- Norm clipping preserves direction when it rescales.
- Value clipping clamps each coordinate and can change direction.
- PyTorch clipping functions modify `.grad` in place.
- Gradient clipping is especially useful for exploding gradients.

---

## References

- PyTorch `clip_grad_norm_`: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad.clip_grad_norm_.html
- PyTorch `clip_grad_value_`: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad.clip_grad_value_.html
- Pascanu, Mikolov, Bengio, "On the difficulty of training recurrent neural networks": https://proceedings.mlr.press/v28/pascanu13.html
