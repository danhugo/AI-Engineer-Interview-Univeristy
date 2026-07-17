# Cosine Annealing with Warm Restarts — Interview Knowledge Sheet

## Intuition

Cosine annealing with warm restarts repeats cosine decay cycles.

Each cycle:

```
start high -> decay smoothly -> restart high
```

The restart can help the optimizer explore again instead of only settling.

---

## 1. The Schedule

Within one cycle:

$$
\text{lr}_t =
\eta_{\min}
+ \frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\frac{\pi T_{cur}}{T_i}\right)\right)
$$

Where:

- `T_cur` is the current position inside the cycle
- `T_i` is the current cycle length
- `eta_min` is the minimum learning rate

At restart, `T_cur` resets to `0`.

---

## 2. Cycle Length

PyTorch uses:

- `T_0`: length of the first cycle
- `T_mult`: multiplier for cycle length after each restart

If:

```
T_0 = 2
T_mult = 2
```

cycle lengths are:

```
2, 4, 8, ...
```

---

## 3. PyTorch Pattern

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=0.0,
)
```

---

## 4. Interview Gotchas

- Warm restarts reset the learning rate high.
- `T_0` controls the first cycle length.
- `T_mult` controls whether cycles grow.
- This is different from plain `CosineAnnealingLR`, which does not restart.
- Restarts are useful when you want repeated exploration and settling.

---

## References

- PyTorch `CosineAnnealingWarmRestarts`: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
- SGDR paper: https://arxiv.org/abs/1608.03983
