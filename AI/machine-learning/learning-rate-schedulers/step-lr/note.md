# StepLR — Interview Knowledge Sheet

## Intuition

StepLR drops the learning rate by a fixed factor every fixed number of steps or epochs.

It is like saying:

```
train at this speed for a while
then slow down suddenly
repeat
```

The drop is abrupt, not smooth.

---

## 1. The Schedule

With base learning rate `lr`, drop factor `gamma`, and `step_size`:

$$
\text{lr}_t = \text{base\_lr}\cdot \gamma^{\left\lfloor t / \text{step\_size}\right\rfloor}
$$

Example:

```
base_lr = 0.1
step_size = 3
gamma = 0.1
```

Learning rates:

```
0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001
```

---

## 2. When to Use

Use StepLR when you want simple scheduled drops.

It is easy to explain and debug.

It is less flexible than smooth schedules like cosine decay.

---

## 3. PyTorch Pattern

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1,
)
```

Common training order:

```python
optimizer.step()
scheduler.step()
```

---

## 4. Interview Gotchas

- StepLR makes sudden drops.
- `gamma` is multiplicative.
- `step_size` controls how often drops happen.
- Call `scheduler.step()` at the right frequency.
- If stepping per batch, `step_size` means batches. If stepping per epoch, it means epochs.

---

## References

- PyTorch `StepLR`: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
