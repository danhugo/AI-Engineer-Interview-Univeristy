# Model Memory Footprint - Interview Knowledge Sheet

## Intuition

Model memory is mostly counting tensors times bytes per element.

For inference, the minimum weight memory is:

$$
\text{parameters} \times \text{bytes per parameter}
$$

Real memory can be higher because frameworks also hold buffers, temporary activations, key-value caches, allocator padding, and sometimes multiple copies during loading.

---

## 1. Common Dtypes

Typical storage sizes:

| dtype | bytes per parameter |
| --- | ---: |
| float32 | 4 |
| bfloat16 | 2 |
| float16 | 2 |
| int8 | 1 |
| int4 | 0.5 |

So a 7B parameter model needs about:

$$
7 \times 10^9 \times 2 = 14 \times 10^9
$$

bytes for float16 weights, before overhead.

---

## 2. Decimal GB vs Binary GiB

Vendors often use decimal gigabytes:

$$
1\ \text{GB} = 10^9\ \text{bytes}
$$

Operating systems and low-level memory tools often report gibibytes:

$$
1\ \text{GiB} = 2^{30}\ \text{bytes}
$$

The same byte count looks smaller in GiB than in GB.

---

## 3. Training Memory

Training needs more than weights:

- parameters
- gradients
- optimizer state
- saved activations

Adam-style optimizers commonly keep two extra state tensors per parameter. In mixed precision training, there may also be master weights.

A simple lower-bound estimate for Adam training in float32 is:

$$
\text{weights} + \text{gradients} + \text{Adam states}
$$

$$
= 4P + 4P + 8P = 16P\ \text{bytes}
$$

Activation memory can dominate for long sequences and large batches.

---

## 4. KV Cache for Autoregressive Inference

Decoder-only LLM inference often stores key and value tensors for past tokens.

A simplified KV cache estimate:

$$
2 \cdot L \cdot B \cdot T \cdot H \cdot \text{bytes}
$$

where:

- `2`: key and value
- `L`: number of layers
- `B`: batch size
- `T`: cached sequence length
- `H`: hidden size across all heads

This cache grows linearly with sequence length.

---

## 5. Quantization

Quantization reduces weight storage by using fewer bits per parameter, such as int8 or int4.

It usually does not make every memory component shrink by the same factor. Activations, KV cache, scales, metadata, and some compute paths may use higher precision.

---

## Interview Gotchas

- Parameters alone are only a lower bound.
- Use bytes per element, not bits, unless you convert carefully.
- Distinguish GB from GiB.
- Training memory includes gradients, optimizer states, and activations.
- Inference memory for long contexts includes KV cache.
- Quantized model memory includes overhead beyond raw bits.

---

## References

- Hugging Face Accelerate model memory estimator: https://huggingface.co/docs/accelerate/main/usage_guides/model_size_estimator
- Hugging Face Transformers quantization docs: https://huggingface.co/docs/transformers/main/quantization
- PyTorch module dtype/device behavior: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
