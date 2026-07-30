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

## Why Use Different Frequencies?

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

## Why $10000$?

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

## RoPE

Video: https://www.youtube.com/watch?v=o29P0Kpobz0

RoPE means Rotary Positional Embedding.

Instead of adding a position vector to token embeddings, RoPE rotates query and
key vectors by a position-dependent angle.

For each pair of dimensions:

$$
[x_{2i}, x_{2i+1}]
$$

RoPE rotates the pair by:

$$
pos \cdot w_i
$$

So nearby positions get small rotation differences. Far positions get larger
rotation differences.

Intuition: position becomes part of the angle, not an added vector.

RoPE is usually applied to $Q$ and $K$, not $V$.

Reason:

$$
\text{Attention score} = QK^\top
$$

Attention decides which tokens match by comparing queries and keys. If RoPE
rotates $Q$ and $K$, the attention score depends on token distance.

So RoPE gives attention relative position information.

Example:

- token at position $m$ gets rotation angle $m \cdot w_i$
- token at position $n$ gets rotation angle $n \cdot w_i$
- their score depends on the difference $(m - n)$

This is why RoPE works well for decoder-only LLMs.
