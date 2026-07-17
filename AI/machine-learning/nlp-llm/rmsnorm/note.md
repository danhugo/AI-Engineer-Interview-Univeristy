# RMSNorm - Interview Knowledge Sheet

## Intuition

RMSNorm is like layer norm without subtracting the mean.

It rescales a hidden vector by its root mean square:

```
x -> divide by RMS(x) -> learned scale
```

The idea is to keep activation scale controlled while avoiding the mean-centering step.

---

## 1. Formula

For one vector $x \in \mathbb{R}^{H}$:

$$
\operatorname{RMS}(x) = \sqrt{\epsilon + \frac{1}{H}\sum_{i=1}^{H}x_i^2}
$$

RMSNorm computes:

$$
y_i = \gamma_i \frac{x_i}{\operatorname{RMS}(x)}
$$

Many implementations do not include a learned bias term.

---

## 2. Difference from LayerNorm

Layer norm:

$$
y_i = \gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

RMSNorm:

$$
y_i = \gamma_i \frac{x_i}{\sqrt{\frac{1}{H}\sum_j x_j^2 + \epsilon}}
$$

LayerNorm recenters and rescales. RMSNorm mostly rescales.

---

## 3. Why It Is Used

RMSNorm is cheaper to compute because it avoids calculating and subtracting the mean.

It is popular in LLM architectures because transformer training often benefits from stable activation scale, and RMSNorm provides that with a simple operation.

---

## 4. Axis Behavior

For sequence data shaped `(B, T, H)`, RMSNorm is usually applied over the hidden dimension.

Each token gets its own RMS value:

$$
\operatorname{RMS}(X_{b,t,:})
$$

The scale parameter `gamma` has shape `(H,)` and broadcasts over batch and sequence.

---

## 5. Epsilon

`epsilon` avoids division by zero and improves numerical stability:

$$
\sqrt{\epsilon + \operatorname{mean}(x^2)}
$$

Without epsilon, an all-zero vector would cause division by zero.

---

## Interview Gotchas

- RMSNorm does not subtract the mean.
- It divides by root mean square, not standard deviation.
- It usually has a learned scale but no bias.
- It is commonly applied over the hidden dimension.
- It controls scale but does not force zero mean.

---

## References

- Zhang and Sennrich, "Root Mean Square Layer Normalization": https://arxiv.org/abs/1910.07467
- NeurIPS 2019 RMSNorm proceedings page: https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html
- PyTorch `RMSNorm`: https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
