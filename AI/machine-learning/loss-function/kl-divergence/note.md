# KL Divergence — Interview Knowledge Sheet

## Intuition

KL divergence measures how different one probability distribution is from another.

It asks:

```
How much information is lost if we use q to approximate p?
```

It is not symmetric.

Usually:

$$
D_{KL}(p \| q) \ne D_{KL}(q \| p)
$$

---

## 1. Formula

For discrete distributions:

$$
D_{KL}(p \| q) = \sum_i p_i\log\left(\frac{p_i}{q_i}\right)
$$

`p` is the target distribution.

`q` is the predicted or approximate distribution.

The best value is `0`, when the distributions match.

---

## 2. Cross-Entropy Connection

KL and cross-entropy are related:

$$
H(p, q) = H(p) + D_{KL}(p \| q)
$$

If `p` is fixed, minimizing cross-entropy also minimizes KL divergence.

That is why distillation and soft-label training often use KL-style losses.

---

## 3. PyTorch Pattern

PyTorch `KLDivLoss` expects:

- input: log-probabilities
- target: probabilities by default

```python
log_q = torch.nn.functional.log_softmax(logits, dim=1)
loss = torch.nn.KLDivLoss(reduction="batchmean")(log_q, p)
```

`batchmean` is often the mathematically expected reduction.

---

## 4. Interview Gotchas

- KL compares distributions, not class IDs.
- KL is not symmetric.
- KL is zero when the distributions match.
- PyTorch input should usually be log-probabilities.
- Use KL for soft targets, distillation, and distribution matching.

---

## References

- PyTorch `KLDivLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
