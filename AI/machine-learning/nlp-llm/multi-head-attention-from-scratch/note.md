# Multi-Head Attention From Scratch - Interview Knowledge Sheet

## Intuition

Attention lets each token build a new representation by looking at other tokens.

The Transformer paper defines scaled dot-product attention as:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Multi-head attention runs this several times in parallel with different learned projections.

Each head gets its own view of the sequence. One head may focus on nearby syntax, another on long-range agreement, another on delimiter-like tokens.

---

## 1. Query, Key, Value

For each token representation $x$:

- query $q$: what this token is looking for
- key $k$: what this token offers to be matched against
- value $v$: what information this token contributes if selected

Scores come from query-key similarity:

$$
s_{ij} = q_i \cdot k_j
$$

Softmax converts scores into weights, and the output is a weighted sum of values.

---

## 2. Why Scale by $\sqrt{d_k}$?

If query and key dimensions are large, dot products can grow in magnitude.

Large logits push softmax into saturated regions where gradients are small.

The Transformer divides by $\sqrt{d_k}$:

$$
\frac{QK^\top}{\sqrt{d_k}}
$$

This keeps score magnitudes more stable as head dimension changes.

---

## 3. Multi-Head Shape Flow

Assume:

- batch size $b$
- sequence length $n$
- model dimension $d_\text{model}$
- number of heads $h$
- head dimension $d_h = d_\text{model} / h$

Typical shape flow:

```
x:          (b, n, d_model)
project:    q, k, v -> (b, n, d_model)
split:      (b, h, n, d_h)
attention:  (b, h, n, d_h)
combine:    (b, n, d_model)
output W_o: (b, n, d_model)
```

---

## 4. Causal Masking

Decoder-only language models use causal self-attention.

Token $i$ may attend to tokens $j \leq i$, but not future tokens:

$$
\text{mask}_{ij} = \mathbb{1}[j \leq i]
$$

This is what makes next-token prediction valid during training.

---

## 5. Why Multiple Heads?

A single attention distribution has one pattern per token.

Multiple heads give the model several attention distributions at the same layer.

The heads are concatenated and mixed by an output projection, so the layer can combine different relationship types.

---

## Interview Gotchas

- Attention weights multiply $V$, not $K$.
- The softmax is over keys for each query.
- The scale is based on per-head key dimension $d_k$, not full model dimension.
- Multi-head attention usually keeps total model dimension fixed and splits it across heads.
- A mask changes which keys are allowed before softmax.
- Self-attention means $Q$, $K$, and $V$ come from the same sequence; cross-attention uses queries from one sequence and keys/values from another.

---

## References

- Vaswani et al., "Attention Is All You Need": https://arxiv.org/abs/1706.03762
- The Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
