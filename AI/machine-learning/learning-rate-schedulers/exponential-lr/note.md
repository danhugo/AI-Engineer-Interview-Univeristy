# ExponentialLR — Interview Knowledge Sheet

## Intuition

ExponentialLR decays the learning rate by the same factor every step.

Instead of sudden occasional drops, it shrinks continuously:

```
lr, lr*gamma, lr*gamma^2, lr*gamma^3, ...
```

It is a smooth multiplicative decay.

---

## 1. The Schedule

With base learning rate `base_lr` and decay factor `gamma`:

$$
\text{lr}_t = \text{base\_lr}\cdot \gamma^t
$$

If `gamma < 1`, the learning rate decays.

If `gamma = 1`, it stays constant.

---

## 2. When to Use

Use ExponentialLR when you want steady decay.

It is simpler than cosine schedules.

But it can decay too quickly if `gamma` is too small.

---

## 3. PyTorch Pattern

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95,
)
```

Common order:

```python
optimizer.step()
scheduler.step()
```

---

## 4. Interview Gotchas

- `gamma` is multiplied every scheduler step.
- Small changes in `gamma` compound over many steps.
- `gamma < 1` decays.
- `gamma > 1` increases the learning rate.
- Know whether you call it per epoch or per batch.

---

## References

- PyTorch `ExponentialLR`: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
