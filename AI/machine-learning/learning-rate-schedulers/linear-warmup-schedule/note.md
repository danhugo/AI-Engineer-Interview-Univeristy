# Linear Warmup Schedule — Interview Knowledge Sheet

## Intuition

Linear warmup starts training with a small learning rate and increases it gradually.

Early in training, weights are random and gradients can be noisy.

Jumping straight to the full learning rate can make training unstable.

Warmup says:

```
start cautious -> ramp up -> train normally
```

---

## 1. The Schedule

For `warmup_steps` steps, increase from `0` to the base learning rate.

$$
\text{lr}_t = \text{base\_lr}\cdot\frac{t}{\text{warmup\_steps}}
$$

After warmup, keep the base learning rate:

$$
\text{lr}_t = \text{base\_lr}
$$

This is often used before another decay schedule.

---

## 2. Why Warmup Helps

Warmup is common in deep learning because the first updates can be risky.

At the start:

- activations are not well calibrated
- optimizer moments are not stable yet
- large updates can push training into a bad region

Warmup gives the model a few steps to settle before full-speed training.

---

## 3. PyTorch Pattern

Use `LambdaLR` for a custom warmup multiplier:

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

Call scheduler after the optimizer step:

```python
optimizer.step()
scheduler.step()
```

---

## 4. Interview Gotchas

- Warmup changes the learning rate, not the optimizer formula.
- It is usually step-based, not epoch-based, in large deep learning runs.
- Too short warmup may not stabilize training.
- Too long warmup can waste training steps.
- Warmup is often combined with cosine or linear decay.

---

## References

- PyTorch `LambdaLR`: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
- Hugging Face scheduler docs: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
