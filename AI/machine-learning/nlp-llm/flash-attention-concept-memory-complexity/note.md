# FlashAttention Concept and Memory Complexity - Interview Knowledge Sheet

## Intuition

Standard attention usually materializes the full attention matrix:

$$
S = \frac{QK^\top}{\sqrt{d}}
$$

For long sequences, $S$ has $n^2$ elements.

FlashAttention keeps the exact attention result but changes how it is computed.

The key idea is IO-awareness: avoid repeatedly reading and writing huge intermediate matrices in slow GPU high-bandwidth memory (HBM). Instead, process blocks that fit in faster on-chip SRAM and use an online softmax to combine partial results safely.

---

## 1. Standard Attention Memory

For one batch/head:

- $Q, K, V$: each $n \times d$
- scores $S$: $n \times n$
- probabilities $P$: $n \times n$
- output $O$: $n \times d$

The problematic pieces are $S$ and $P$:

$$
O(n^2)
$$

This becomes large quickly as context length grows.

---

## 2. FlashAttention Is Exact

FlashAttention is not an approximation to softmax attention.

It computes the same mathematical result:

$$
O = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

but avoids storing the full $n \times n$ attention matrix.

---

## 3. Tiling

FlashAttention splits $Q$, $K$, and $V$ into blocks.

For a block of queries, it streams through blocks of keys and values.

At each step, it computes a local score block, updates the running softmax statistics, and updates the output block.

The score block is small enough to live in fast memory.

---

## 4. Online Softmax

Softmax needs all logits because:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

FlashAttention handles this block by block by keeping:

- running max $m$
- running normalizer $\ell$
- running weighted value sum

When a new block arrives, the max and normalizer are rescaled so previous blocks and new blocks share the same denominator.

For old max $m$, old normalizer $\ell$, and new logits block $x$:

$$
m' = \max(m, \max(x))
$$

$$
\ell' = e^{m-m'}\ell + \sum_j e^{x_j-m'}
$$

The output numerator is rescaled the same way.

---

## 5. Backward Pass Idea

Standard attention may save the full probability matrix for backward.

FlashAttention can recompute attention blocks during backward instead of storing the full matrix.

This trades some compute for much lower memory traffic.

---

## 6. Complexity Summary

Math compute remains quadratic in sequence length for dense attention:

$$
O(n^2d)
$$

But attention memory can drop from quadratic to roughly linear in sequence length:

$$
O(n^2) \rightarrow O(nd)
$$

The practical speedup comes from reducing HBM reads/writes, not from changing the dense attention math into subquadratic math.

---

## Interview Gotchas

- FlashAttention is exact attention, not sparse or approximate attention.
- It reduces memory traffic and activation memory by tiling and recomputation.
- Dense attention compute is still $O(n^2d)$.
- The main bottleneck addressed is IO between GPU memory levels.
- Online softmax is what makes blockwise exact softmax possible.
- Benefits grow with longer sequences because the avoided attention matrix is $n \times n$.

---

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness": https://arxiv.org/abs/2205.14135
- Official FlashAttention repository: https://github.com/Dao-AILab/flash-attention
