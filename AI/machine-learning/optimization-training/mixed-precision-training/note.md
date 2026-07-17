# Mixed Precision Training - Interview Knowledge Sheet

## Intuition

Mixed precision trains with more than one floating-point type.

The usual idea:

- use lower precision, such as `float16` or `bfloat16`, for many forward and backward operations
- keep numerically sensitive work and master weights in `float32`
- use loss scaling when `float16` gradients may underflow

This can reduce memory use and speed up training on hardware designed for low-precision math.

---

## 1. Why It Helps

`float16` uses 16 bits per number instead of 32.

That can help because:

- activations take less memory
- gradients take less memory
- tensor-core hardware can run some matrix multiplications faster

But lower precision has less range and less detail.

So mixed precision is not "make everything half precision." It is "use low precision where it is safe, keep precision where it matters."

---

## 2. FP16 vs BF16

`float16`:

- small memory footprint
- fast on many GPUs
- narrow exponent range
- often needs gradient scaling

`bfloat16`:

- same memory footprint
- wider exponent range, similar to `float32`
- usually less likely to underflow
- has fewer mantissa bits than `float16`

In interviews, say that the best dtype depends on hardware and model stability.

---

## 3. Autocast

Autocast chooses lower precision for operations that are usually safe and keeps other operations in higher precision.

PyTorch pattern:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)
```

Do the forward pass under autocast.

Do not manually cast every parameter to half unless you know exactly why.

---

## 4. Gradient Scaling

Small `float16` gradients can underflow to zero.

Loss scaling multiplies the loss before backpropagation:

$$
L_\text{scaled} = sL
$$

Gradients are scaled too:

$$
\nabla(sL) = s\nabla L
$$

Before the optimizer update, divide gradients by the same scale:

$$
\nabla L = \frac{\nabla(sL)}{s}
$$

This preserves tiny gradients during backward computation.

---

## 5. Master Weights

Optimizers usually keep model parameters or optimizer state in `float32`.

Reason:

- weight updates can be small
- optimizer states such as Adam moments need stable accumulation
- repeated low-precision updates can lose information

Low precision is mainly for compute and stored activations; `float32` remains important for state.

---

## 6. PyTorch AMP Pattern

Typical PyTorch pattern:

```python
scaler = torch.amp.GradScaler("cuda")

for input, target in loader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(input)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

`GradScaler` can skip optimizer steps if gradients contain infinities or NaNs.

---

## Interview Gotchas

- Mixed precision is not the same as pure half-precision training.
- Keep master weights and optimizer state stable, often in `float32`.
- `float16` may need loss scaling because small gradients can underflow.
- Autocast belongs around the forward pass and loss computation.
- Backward under autocast is not the usual recommendation.
- Check for NaNs, overflow, and metric drift when enabling AMP.

---

## References

- PyTorch Automatic Mixed Precision examples: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- PyTorch AMP docs: https://docs.pytorch.org/docs/stable/amp.html
- Micikevicius et al., "Mixed Precision Training": https://arxiv.org/abs/1710.03740
