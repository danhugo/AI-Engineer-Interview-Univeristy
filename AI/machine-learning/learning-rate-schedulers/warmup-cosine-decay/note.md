# Warmup + Cosine Decay — Interview Knowledge Sheet

## Intuition

Warmup plus cosine decay is a two-phase learning rate schedule.

First:

```
increase learning rate slowly
```

Then:

```
decay it smoothly toward a small value
```

This is common in deep learning and transformer training.

Warmup stabilizes early training.

Cosine decay lowers the learning rate later so the model can settle.

---

## 1. Warmup Phase

For the first `warmup_steps`:

$$
\text{lr}_t = \text{base\_lr}\cdot\frac{t}{\text{warmup\_steps}}
$$

The learning rate ramps from `0` to `base_lr`.

---

## 2. Cosine Decay Phase

After warmup, let progress go from `0` to `1`:

$$
\text{progress} =
\frac{t-\text{warmup\_steps}}
{\text{total\_steps}-\text{warmup\_steps}}
$$

Then:

$$
\text{lr}_t =
\text{min\_lr}
+ \frac{1}{2}(\text{base\_lr}-\text{min\_lr})
\left(1 + \cos(\pi\cdot\text{progress})\right)
$$

At the start of decay, the learning rate is near `base_lr`.

At the end, it is near `min_lr`.

---

## 3. Why Cosine?

Cosine decay is smooth.

It drops slowly at first, faster in the middle, and slowly again near the end.

That gives training a gentle landing instead of an abrupt drop.

---

## 4. PyTorch Pattern

Use `LambdaLR` for custom warmup plus cosine:

```python
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

In Hugging Face Transformers, this pattern is available as:

```python
get_cosine_schedule_with_warmup(...)
```

---

## 5. Interview Gotchas

- Warmup helps early stability.
- Cosine decay helps late-stage convergence.
- Use total training steps, not just epochs, for step-based schedules.
- Decide whether final learning rate should reach `0` or `min_lr`.
- Call `scheduler.step()` after `optimizer.step()`.

---

## References

- PyTorch `LambdaLR`: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
- Hugging Face scheduler docs: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
- SGDR paper: https://arxiv.org/abs/1608.03983
