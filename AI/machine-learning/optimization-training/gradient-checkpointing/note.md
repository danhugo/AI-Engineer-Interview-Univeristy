# Gradient Checkpointing - Interview Knowledge Sheet

## Intuition

Backpropagation needs intermediate activations from the forward pass.

For deep networks, those saved activations can use a lot of memory.

Gradient checkpointing saves only selected activations and recomputes the missing ones during backward.

It trades compute for memory:

```
less activation memory
more forward computation during backward
same gradient goal
```

---

## 1. What Normally Happens

During a normal forward pass, autograd saves tensors needed to compute gradients later.

For a chain of layers:

```
x -> layer1 -> layer2 -> layer3 -> loss
```

the backward pass may need outputs from intermediate layers.

Saving all of them is fast for backward but memory-heavy.

---

## 2. Checkpointing Idea

Instead of saving every intermediate activation, choose checkpoints.

During backward:

1. load the nearest saved checkpoint activation
2. rerun the forward operations inside that segment
3. use the recomputed activations to calculate gradients

The model does extra work, but peak memory is lower.

---

## 3. Memory vs Compute Tradeoff

Without checkpointing:

$$
\text{memory} \approx \text{many saved activations}
$$

With checkpointing:

$$
\text{memory} \downarrow
$$

but:

$$
\text{compute} \uparrow
$$

This is useful when training is memory-bound, such as large transformers, long sequences, high-resolution images, or large batch sizes.

---

## 4. PyTorch Pattern

PyTorch provides `torch.utils.checkpoint`.

```python
from torch.utils.checkpoint import checkpoint

out = checkpoint(block, x, use_reentrant=False)
```

`block` should be a function or module segment.

Modern PyTorch recommends passing `use_reentrant` explicitly.

---

## 5. Randomness and Dropout

Checkpointing reruns forward code during backward.

If the segment uses randomness, such as dropout, recomputation must see the same random choices to match ordinary training.

PyTorch preserves RNG state by default, but that has overhead.

If deterministic equivalence is not needed, `preserve_rng_state=False` can reduce overhead.

---

## 6. What It Is Not

Gradient checkpointing is not saving model checkpoints to disk.

It does not mean "save weights every N epochs."

It is activation checkpointing inside one training step.

---

## Interview Gotchas

- The main tradeoff is lower activation memory for extra recomputation.
- It should produce equivalent gradients when used correctly.
- It is most useful when activation memory dominates.
- Dropout and other randomness need careful RNG handling.
- It is different from training checkpoints saved for recovery.
- In PyTorch, pass `use_reentrant` explicitly.

---

## References

- PyTorch `torch.utils.checkpoint`: https://docs.pytorch.org/docs/stable/checkpoint.html
- PyTorch activation checkpointing blog: https://pytorch.org/blog/activation-checkpointing-techniques/
- Chen et al., "Training Deep Nets with Sublinear Memory Cost": https://arxiv.org/abs/1604.06174
