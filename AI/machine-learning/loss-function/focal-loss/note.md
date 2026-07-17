# Focal Loss — Interview Knowledge Sheet

## Intuition

Focal loss is cross-entropy with extra focus on hard examples.

Easy examples already have high probability for the correct class. They do not need to dominate training.

Hard examples have low probability for the correct class. Focal loss gives them more weight.

It is useful for class imbalance, especially dense detection.

---

## 1. Formula

Let `p_t` mean the probability assigned to the true class.

Binary focal loss is:

$$
\text{FL}(p_t) = -\alpha(1-p_t)^\gamma\log(p_t)
$$

Where:

- `alpha` balances classes
- `gamma` focuses on hard examples

If `gamma = 0`, focal loss becomes weighted cross-entropy.

---

## 2. Why It Works

When the model is correct and confident, `p_t` is near `1`.

Then:

$$
(1-p_t)^\gamma
$$

is small, so the easy example is down-weighted.

When the model is wrong, `p_t` is small. The weight stays large.

---

## 3. PyTorch Pattern

PyTorch does not need a special built-in focal loss for basic practice.

Compute binary cross-entropy from logits without reduction:

```python
bce = torch.nn.functional.binary_cross_entropy_with_logits(
    logits, targets, reduction="none"
)
```

Then multiply by the focal weight and average.

---

## 4. Interview Gotchas

- Focal loss is for imbalanced classification.
- It down-weights easy examples.
- `gamma` controls how strongly easy examples are down-weighted.
- `alpha` handles class weighting.
- Use logits in code for numerical stability.

---

## References

- Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
- PyTorch `BCEWithLogitsLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
