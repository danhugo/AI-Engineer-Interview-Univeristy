# Transformer From Scratch

Build Transformer blocks in PyTorch from low-level pieces.

Goal: know the tensor shapes, learned weights, and fixed buffers.

## Embedding Block

The embedding block maps token IDs to vectors.

For vocabulary size $V$ and model dimension $d_{\text{model}}$:

$$
W_E \in \mathbb{R}^{V \times d_{\text{model}}}
$$

For token ID $t$, return row $W_E[t]$.

Input shape:

$$
(\text{batch}, \text{seq\_len})
$$

Output shape:

$$
(\text{batch}, \text{seq\_len}, d_{\text{model}})
$$

Weights are initialized from a normal distribution.

### Padding Index

Use `<PAD>` when sequences in a batch have different lengths.

Example:

```text
It is a cat        -> It is a cat <PAD>
It is a yellow cat -> It is a yellow cat
```

`<PAD>` is not real content, so it should not learn meaning. If `padding_idx` is
provided:

1. The embedding vector at `padding_idx` is initialized to zero.
2. The gradient for that row is forced to zero during backpropagation.
3. The optimizer step does not update that row.

Conceptually:

```text
compute gradients
-> hook clears gradient at padding_idx
-> optimizer updates all non-padding rows
```

Use `register_hook` to clear the padding row gradient before the optimizer step:

$$
\nabla W_E[\text{padding\_idx}] = 0
$$

This keeps the padding vector zero.

## Positional Encoding Block

Self-attention alone does not know order.

Without position information, these look the same:

```text
Hello world
```

or:

```text
world Hello
```

Token embeddings say what the token is. Positional encodings say where it is.

The original Transformer uses fixed sinusoidal positional encodings:

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

Terms:

- $pos$ is the token position in the sequence.
- $i$ is the dimension pair index.
- $d_{\text{model}}$ is the embedding dimension.
- even dimensions use sine.
- odd dimensions use cosine.

Same shape as token embeddings:

$$
(\text{batch}, \text{seq\_len}, d_{\text{model}})
$$

Add token and position:

$$
X_{\text{input}} = X_{\text{token}} + PE
$$

### Why Use Different Frequencies?

The sinusoidal formula can also be written as:

$$
PE(pos, 2i) = \sin(pos \cdot w_i)
$$

where:

$$
w_i = \frac{1}{10000^{2i / d_{\text{model}}}}
$$

$w_i$ means how many radians the wave moves per token step.

Each dimension uses a different $w_i$.

- small $i$ -> large $w_i$ -> changes fast
- large $i$ -> small $w_i$ -> changes slowly

For intuition, one position step is one token step. We imply $v = 1$.

So the model gets both local and long-range position signals.

### Why $10000$?

The value $10000$ controls the wavelength range.

For a wave:

$$
w = \frac{2\pi}{\lambda}
$$

so:

$$
\lambda = \frac{2\pi}{w}
$$

If $w = 1$, then:

$$
\lambda = 2\pi \approx 6.28
$$

One cycle takes about 6 tokens.

For the slowest dimensions, $w$ approaches:

$$
\frac{1}{10000}
$$

so:

$$
\lambda = 2\pi \cdot 10000 \approx 62831
$$

One cycle takes about 62,000 tokens.

So PE mixes fast waves and slow waves. This helps avoid repeated position
patterns inside the context window.

Transformer has d_model = 1024, so this number is large enough to model position of tokens.

### RoPE

Video: https://www.youtube.com/watch?v=o29P0Kpobz0

[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)

RoPE means Rotary Positional Embedding.

#### Why RoPE?

Absolute PE tells the model **where** each token is, but it does not represent
the distance $j-i$ directly. Learned absolute PE stores one trainable vector
for each position, so the table size limits the maximum position. Sinusoidal PE
uses a fixed formula and can generate any position, but the model may still
generalize poorly beyond its training length.

Relative PE tells the model **how far apart** two tokens are. For example, when
position $i$ attends to the previous position $j=i-1$, their relative distance
is always $j-i=-1$, no matter where they appear in the sequence. Relative PE
often adds this information through a bias lookup inside attention. This adds
overhead, can complicate optimized attention kernels, and does not directly
preserve absolute position.

$$
\text{score}(i,j)=\frac{Q_iK_j^\top}{\sqrt{d_k}}+b_{j-i}
$$

Here, $b_{j-i}$ is a learned bias for the relative distance between tokens. For
a given sequence length, these values form a bias matrix $B$ that **depends only
on token positions, not token content**:

$$
\text{Attention scores}=\frac{QK^\top}{\sqrt{d_k}}+B
$$

After training, the bias matrix $B$ is **fixed for the same relative
positions**. Adding it inside attention requires extra work or explicit support
from the fused attention kernel.

#### Core intuition

RoPE groups the dimensions of each $Q$ and $K$ vector into 2D pairs, such as
$[x_{2i},x_{2i+1}]$. At position $p$, it rotates each dimension pair by:

$$
\theta_{p,i}=p \cdot w_i
$$

$w_i$ means how many radians the wave moves per token step. Each rotated vector
therefore **depends on its absolute position**.

#### Why rotation works

For tokens at positions $m$ and $n$, the angle difference is:

$$
\theta_{n,i}-\theta_{m,i}=(n-m)w_i
$$

$R_p$ is the rotation matrix for position $p$. For one 2D pair:

$$
R(\theta_{p,i})=
\begin{bmatrix}
\cos\theta_{p,i} & -\sin\theta_{p,i} \\
\sin\theta_{p,i} & \cos\theta_{p,i}
\end{bmatrix}
$$

The full $R_p$ applies one such rotation to every dimension pair, using a
different $w_i$ for each pair.

In practice, the full matrix $R_p$ **is not built explicitly**. Each pair is
rotated using cached sine and cosine values:

$$
(x_0,x_1)\mapsto
(x_0\cos\theta-x_1\sin\theta,\;
 x_0\sin\theta+x_1\cos\theta)
$$

For $n$ tokens with $d$ dimensions each, RoPE touches every dimension once.
Applying it to both $Q$ and $K$ **costs** $O(2nd)=O(nd)$. In contrast, attention
compares every pair of tokens, so $QK^\top$ **costs** $O(n^2d)$. The RoPE
rotation is therefore small in comparison.

The same difference appears in their attention score:

$$
\left(R_m Q_m\right)^\top\left(R_n K_n\right)
=Q_m^\top R_{n-m}K_n
$$

Therefore, **RoPE uses absolute positions for rotations, while attention scores
depend on relative distance**. After rotating $Q$ and $K$, attention still uses
the usual $QK^\top$ computation, so **RoPE does not require an extra bias matrix
inside the attention kernel**. RoPE is applied to $Q$ and $K$, not $V$, because
$QK^\top$ computes attention scores.

#### Distance effect

Nearby positions have small rotation differences. As distance grows, the
different 2D pairs tend to become less aligned, so positional correlation often
weakens. **This is not a strict rule:** content can still make distant tokens
attend strongly.

#### Why RoPE fits self-attention

In self-attention, $Q$ and $K$ come from the same sequence, so their position
difference $j-i$ is meaningful. RoPE therefore works naturally in both encoder
and decoder self-attention.

For autoregressive decoding, RoPE also works efficiently with the KV cache:
each key is **rotated once before being cached**, while each new query is rotated
at its current position. Cross-attention needs more care because its queries and
keys come from different sequences, so $j-i$ may not be meaningful.

## Multihead Attention Block
