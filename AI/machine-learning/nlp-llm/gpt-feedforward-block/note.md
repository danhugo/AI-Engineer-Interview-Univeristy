# GPT Feed-Forward Block - Interview Knowledge Sheet

## Intuition

In a transformer block, attention mixes information across tokens. The feed-forward network then transforms each token independently.

For every token vector:

```
hidden size -> wider inner size -> nonlinearity -> hidden size
```

The same feed-forward weights are applied to every sequence position.

---

## 1. Position-Wise MLP

The original Transformer describes a position-wise feed-forward network:

$$
\operatorname{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

GPT-style models often use GELU rather than ReLU:

$$
\operatorname{FFN}(x) = \operatorname{GELU}(xW_1 + b_1)W_2 + b_2
$$

The operation is position-wise because token `t` is transformed without directly reading token `t + 1` or token `t - 1`.

---

## 2. Shapes

For `X` shaped `(B, T, D)`:

- `D`: model hidden size
- `M`: feed-forward inner size, often around `4D`

Common weight shapes are:

- `W1`: `(D, M)`
- `b1`: `(M,)`
- `W2`: `(M, D)`
- `b2`: `(D,)`

Output shape remains `(B, T, D)` so the result can be added back to the residual stream.

---

## 3. Why Expand Then Project Back

The hidden vector first moves into a wider feature space where the nonlinearity can create richer feature interactions. The second projection compresses it back to the model width.

The block is not mixing sequence positions. Attention handles token mixing; the feed-forward block handles per-token feature transformation.

---

## 4. Parameter Count

Ignoring biases:

$$
D \cdot M + M \cdot D = 2DM
$$

With biases:

$$
D \cdot M + M + M \cdot D + D
$$

If $M = 4D$, the feed-forward weights have about:

$$
8D^2
$$

parameters per layer.

---

## 5. Residual Placement

A GPT block usually wraps the MLP with normalization and a residual connection:

```python
x = x + mlp(norm(x))
```

The MLP must return the same hidden size as its input. Otherwise the residual addition is invalid.

---

## Interview Gotchas

- The feed-forward block is shared across positions.
- It changes features within each token, not token-to-token attention.
- Output shape must match input hidden size for residual addition.
- The inner dimension is commonly larger than the model dimension.
- GPT-style implementations commonly use GELU-family activations.

---

## References

- Vaswani et al., "Attention Is All You Need": https://arxiv.org/abs/1706.03762
- OpenAI GPT-2 repository: https://github.com/openai/gpt-2
- PyTorch `TransformerEncoderLayer`: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
