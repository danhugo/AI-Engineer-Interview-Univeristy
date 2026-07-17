# CosineAnnealingLR — Interview Knowledge Sheet

## Intuition

Cosine annealing lowers the learning rate smoothly from a high value to a low value.

It starts slow, drops faster in the middle, then slows again near the end.

This gives training a gentle landing.

---

## 1. The Schedule

PyTorch's closed-form intuition is:

$$
\text{lr}_t =
\eta_{\min}
+ \frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\frac{\pi t}{T_{\max}}\right)\right)
$$

Where:

- `eta_max` is the starting learning rate
- `eta_min` is the minimum learning rate
- `T_max` is the number of steps in the decay

---

## 2. Why Cosine?

Cosine decay is smooth.

It avoids sudden drops like StepLR.

It is often used when you know the training budget.

---

## 3. PyTorch Pattern

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=0.0,
)
```

PyTorch's `CosineAnnealingLR` does not restart. Use `CosineAnnealingWarmRestarts` for restarts.

---

## 4. Interview Gotchas

- Cosine annealing decays smoothly.
- `T_max` should match the intended cycle length.
- `eta_min` is the floor learning rate.
- This scheduler does not restart by itself.
- Call `scheduler.step()` after `optimizer.step()`.

---

## References

- PyTorch `CosineAnnealingLR`: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
- SGDR paper: https://arxiv.org/abs/1608.03983
