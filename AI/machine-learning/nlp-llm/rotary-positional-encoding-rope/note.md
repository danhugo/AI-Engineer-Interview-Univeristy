# Rotary Positional Encoding RoPE - Interview Knowledge Sheet

## Intuition

Transformers need position information because attention alone is permutation-invariant.

RoPE, introduced in RoFormer, injects position by rotating query and key vectors.

Instead of adding a position vector to token embeddings, RoPE rotates pairs of dimensions by an angle that depends on token position.

The important result: after rotating $q$ at position $m$ and $k$ at position $n$, their dot product naturally depends on the relative offset $m-n$.

---

## 1. Rotation in Pairs

For a pair of dimensions $(x_{2i}, x_{2i+1})$, RoPE applies a 2D rotation:

$$
\begin{bmatrix}
x'_{2i} \\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
$$

where $m$ is the token position and $\theta_i$ is a frequency for that dimension pair.

Commonly:

$$
\theta_i = 10000^{-2i/d}
$$

---

## 2. Efficient Vector Form

Implementations usually avoid building rotation matrices.

They compute:

$$
\text{RoPE}(x,m) = x \odot \cos(m\theta) + \text{rotate\_half}(x) \odot \sin(m\theta)
$$

For each pair:

$$
\text{rotate\_half}([a,b]) = [-b,a]
$$

---

## 3. Why Apply RoPE to Q and K?

Attention scores use $QK^\top$.

RoPE is designed so positional information affects the query-key dot product.

Values usually are not rotated, because values are the content being mixed after attention weights are computed.

---

## 4. Absolute and Relative Behavior

Each token gets an absolute position-specific rotation.

But because dot products between rotated vectors depend on the difference between positions, attention receives relative-position structure.

This is why RoPE is popular in decoder-only LLMs.

---

## 5. Useful Properties

- no learned position table is required
- preserves vector norms because rotation is length-preserving
- gives attention scores relative-position information
- can be extended to long contexts with frequency scaling methods
- works cleanly with cached keys in autoregressive decoding

---

## Interview Gotchas

- RoPE is usually applied to queries and keys, not values.
- The dimension must be even, because dimensions are rotated in pairs.
- It rotates, it does not concatenate or add a learned position embedding.
- Dot products gain relative-position dependence even though rotations use absolute positions.
- Rotation preserves vector norm.

---

## References

- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding": https://arxiv.org/abs/2104.09864
- RoFormer official repository: https://github.com/ZhuiyiTechnology/roformer
- Hugging Face RoFormer docs: https://huggingface.co/docs/transformers/model_doc/roformer
