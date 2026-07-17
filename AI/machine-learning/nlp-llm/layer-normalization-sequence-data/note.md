# Layer Normalization for Sequence Data - Interview Knowledge Sheet

## Intuition

Layer normalization normalizes each token representation by looking across its hidden features.

For sequence tensors shaped `(batch, sequence, hidden)`, it usually leaves `batch` and `sequence` alone and normalizes the last dimension:

```
one token vector -> mean/variance over hidden features -> scale and shift
```

This makes it naturally useful for transformers, where sequence positions can be processed independently and batch size may vary.

---

## 1. Formula

For one hidden vector $x \in \mathbb{R}^{H}$:

$$
\mu = \frac{1}{H}\sum_{i=1}^{H}x_i
$$

$$
\sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Then apply learned affine parameters:

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

`gamma` and `beta` usually have shape `(hidden,)`.

---

## 2. Sequence Axis Behavior

For `X` shaped `(B, T, H)`:

- `B`: batch size
- `T`: sequence length
- `H`: hidden size

Layer norm computes a separate mean and variance for each `(batch, token)` pair over the `H` features.

So token 3 in example 0 is normalized using only token 3's hidden vector, not other tokens and not other batch examples.

---

## 3. Layer Norm vs Batch Norm

Batch norm depends on batch statistics. Layer norm depends on each example's own features.

That matters for sequence models because:

- sequence lengths can vary
- batch sizes can be small
- autoregressive inference may process one example or one token at a time
- train and eval behavior should stay consistent

Layer norm uses input statistics in both training and evaluation.

---

## 4. Why Transformers Use It

Transformers combine attention, feed-forward blocks, residual connections, and normalization.

Layer norm keeps residual streams in a numerically manageable range while preserving per-token independence. Modern transformer variants differ on placement:

- post-norm: `x + sublayer(x)` then normalize
- pre-norm: normalize first, then apply sublayer and residual add

The implementation details affect optimization, but the core normalization formula is the same.

---

## 5. Epsilon and Affine Parameters

`epsilon` prevents division by zero when variance is tiny:

$$
\sqrt{\sigma^2 + \epsilon}
$$

The affine parameters let the model restore useful scales and offsets after normalization. Without them, every normalized token vector would be forced toward zero mean and unit variance.

---

## Interview Gotchas

- For `(B, T, H)`, normalize over `H`, not over `B` or `T`.
- Layer norm has the same behavior in train and eval modes.
- It has learnable per-hidden-feature `gamma` and `beta` when affine parameters are enabled.
- It uses a population-style variance in common libraries.
- It is different from batch norm, which uses batch-dependent statistics.

---

## References

- Ba, Kiros, and Hinton, "Layer Normalization": https://arxiv.org/abs/1607.06450
- PyTorch `LayerNorm`: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- Vaswani et al., "Attention Is All You Need": https://arxiv.org/abs/1706.03762
