# Sparse Window Attention - Interview Knowledge Sheet

## Intuition

Full self-attention lets every token look at every other token.

For sequence length $n$, that creates an $n \times n$ score matrix:

$$
\text{scores} = \frac{QK^\top}{\sqrt{d_k}}
$$

Sparse window attention keeps the same attention idea, but removes most edges.

Instead of:

```
token i attends to all tokens
```

it uses:

```
token i attends only to nearby tokens
```

This is useful when local context carries most of the signal, or when long sequences make full attention too expensive.

---

## 1. Full Attention Cost

For one attention head:

- $Q, K, V \in \mathbb{R}^{n \times d}$
- score matrix has shape $n \times n$
- compute cost is roughly $O(n^2 d)$
- attention weight memory is roughly $O(n^2)$

The quadratic part comes from comparing every query token with every key token.

---

## 2. Windowed Attention

With a window radius $w$, token $i$ attends only to keys $j$ where:

$$
|i - j| \leq w
$$

For causal language modeling, token $i$ also cannot attend to future keys:

$$
j \leq i
$$

So the causal local window is:

$$
i - w \leq j \leq i
$$

Now each token attends to at most $2w + 1$ keys for bidirectional attention, or $w + 1$ keys for causal attention.

If $w$ is much smaller than $n$, the useful work is closer to:

$$
O(nwd)
$$

---

## 3. The Mask View

Sparse attention is often implemented as a mask over the usual attention scores.

Valid positions keep their score. Invalid positions get a very negative value before softmax:

$$
\text{masked_scores}_{ij} =
\begin{cases}
\text{score}_{ij}, & \text{if } j \text{ is allowed} \\
-\infty, & \text{otherwise}
\end{cases}
$$

Then:

$$
\text{Attention}(Q,K,V) = \text{softmax}(\text{masked_scores})V
$$

The math is the same as full attention, but the connectivity pattern is sparse.

---

## 4. Why It Helps

Sparse window attention can reduce memory and compute when the implementation avoids materializing the full dense matrix.

A dense masked implementation is easier to understand, but it still builds the $n \times n$ scores.

A real optimized sparse implementation computes only the allowed blocks or bands.

---

## 5. Tradeoffs

Window attention is efficient, but it limits direct information flow.

If token 1 needs token 10,000, a fixed small window cannot connect them in one layer.

Common fixes:

- stack many layers so information propagates gradually
- add global tokens
- use dilated or strided attention patterns
- combine local attention with occasional full attention
- use retrieval or memory mechanisms

---

## Interview Gotchas

- Sparse window attention changes the attention graph, not the definition of softmax attention on allowed keys.
- Dense masking is conceptually correct but may not save memory.
- Causal window attention uses only past keys inside the window.
- Bidirectional window attention can look left and right.
- It improves long-context efficiency at the cost of weaker direct long-range interactions.

---

## References

- Vaswani et al., "Attention Is All You Need": https://arxiv.org/abs/1706.03762
- Beltagy et al., "Longformer: The Long-Document Transformer": https://arxiv.org/abs/2004.05150
- Child et al., "Generating Long Sequences with Sparse Transformers": https://arxiv.org/abs/1904.10509
