# LoRA / QLoRA Parameter Count - Interview Knowledge Sheet

## Intuition

Full fine-tuning updates every weight in a model.

LoRA freezes the original weight matrix and learns a small low-rank update instead.

For a linear layer:

$$
W \in \mathbb{R}^{d_{out} \times d_{in}}
$$

LoRA writes the update as:

$$
\Delta W = BA
$$

where:

$$
B \in \mathbb{R}^{d_{out} \times r}
$$

$$
A \in \mathbb{R}^{r \times d_{in}}
$$

The rank `r` is much smaller than `d_in` or `d_out`.

---

## 1. Parameter Count

The frozen base layer has:

$$
d_{out}d_{in}
$$

parameters.

The LoRA adapter has:

$$
r d_{in} + r d_{out}
$$

trainable parameters.

So:

$$
\text{LoRA params} = r(d_{in} + d_{out})
$$

If the layer has a bias, LoRA usually does not add a second bias unless configured.

---

## 2. Example

For a `4096 x 4096` projection and `r = 8`:

Base parameters:

$$
4096 \cdot 4096 = 16{,}777{,}216
$$

LoRA parameters:

$$
8(4096 + 4096) = 65{,}536
$$

Trainable fraction:

$$
\frac{65{,}536}{16{,}777{,}216} \approx 0.39\%
$$

That is the core parameter-efficiency story.

---

## 3. Scaling

LoRA usually applies the update with a scale:

$$
h = xW^T + \frac{\alpha}{r}x(BA)^T
$$

`alpha` changes update magnitude. It does not change parameter count.

Rank `r` changes both capacity and parameter count.

---

## 4. Which Layers Count?

In transformers, LoRA is often attached to attention projections:

- query projection
- key projection
- value projection
- output projection

It may also be attached to feedforward projections.

Total LoRA parameters are the sum over adapted linear layers:

$$
\sum_{\ell} r_\ell(d_{in,\ell} + d_{out,\ell})
$$

Always specify which modules are adapted.

---

## 5. QLoRA

QLoRA combines a frozen quantized base model with trainable LoRA adapters.

The base model weights are stored in low precision, commonly 4-bit. Gradients flow through the frozen quantized model into the LoRA parameters.

Important distinction:

- LoRA reduces trainable parameters.
- QLoRA reduces memory for the frozen base model.

The trainable parameter count is still the LoRA adapter count.

---

## 6. Memory Estimate

A rough training memory estimate for QLoRA-style adapter tuning:

$$
\text{base bytes} \approx \frac{\text{base params} \cdot \text{base bits}}{8}
$$

$$
\text{adapter bytes} \approx \text{LoRA params} \cdot \text{adapter bytes per param}
$$

Optimizer state can dominate adapter memory. Adam commonly stores two moment buffers, so adapter optimizer memory is roughly:

$$
2 \cdot \text{LoRA params} \cdot \text{state bytes}
$$

Real systems also need activations, gradients, temporary buffers, quantization metadata, and framework overhead.

---

## Interview Gotchas

- LoRA freezes `W` and trains `A` and `B`.
- For a linear layer, LoRA trainable params are `r * (in_features + out_features)`.
- `alpha` changes scaling, not parameter count.
- QLoRA does not make the LoRA adapter 4-bit by default; it quantizes the frozen base model.
- Always say which layers receive adapters.
- Full memory is more than parameter storage because optimizer states and activations matter.

---

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models": https://openreview.net/forum?id=nZeVKeeFYf9
- Microsoft `loralib` implementation: https://github.com/microsoft/LoRA
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs": https://arxiv.org/abs/2305.14314
