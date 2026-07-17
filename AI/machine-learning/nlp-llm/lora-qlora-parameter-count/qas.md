# LoRA / QLoRA Parameter Count - Q&A

---

## Intuition

**Q: What does LoRA freeze?**
A: The pretrained base weights.

**Q: What does LoRA train?**
A: Low-rank adapter matrices.

**Q: Why is LoRA parameter-efficient?**
A: It trains `A` and `B` instead of the full weight matrix.

---

## 1. Parameter Count

**Q: For a linear layer with shape `(out_features, in_features)`, how many base parameters are there?**
A: `out_features * in_features`.

**Q: What is the LoRA trainable parameter count?**
A: `rank * (in_features + out_features)`.

**Q: Does LoRA alpha change parameter count?**
A: No.

---

## 2. Transformer Layers

**Q: Which transformer modules often receive LoRA adapters?**
A: Attention projections such as query, key, value, and output projections.

**Q: How do you count total LoRA parameters across a model?**
A: Sum `rank * (in_features + out_features)` over adapted layers.

**Q: Why must you specify target modules?**
A: Different target modules can change adapter count by a large factor.

---

## 3. QLoRA

**Q: What does QLoRA quantize?**
A: The frozen base model.

**Q: Does QLoRA primarily reduce trainable parameter count or base-model memory?**
A: Base-model memory.

**Q: Are LoRA adapter weights necessarily 4-bit in QLoRA?**
A: No.

---

## 4. Memory

**Q: What memory is missing from a simple parameter-count estimate?**
A: Activations, gradients, optimizer states, temporary buffers, and quantization metadata.

**Q: Why can Adam optimizer state matter for LoRA?**
A: Adam stores moment buffers for trainable adapter parameters.

**Q: What is the key interview distinction between LoRA and QLoRA?**
A: LoRA reduces trainable parameters; QLoRA stores the frozen base model in low precision.
