# Model Memory Footprint - Q&A

---

## Basics

**Q: What is the simplest estimate for model weight memory?**
A: `num_parameters * bytes_per_parameter`.

**Q: How many bytes is float32?**
A: 4 bytes.

**Q: How many bytes is float16 or bfloat16?**
A: 2 bytes.

---

## Units

**Q: What is one decimal GB?**
A: `1,000,000,000` bytes.

**Q: What is one GiB?**
A: `2^30` bytes.

**Q: Why can memory numbers differ between tools?**
A: Some report GB and others report GiB, and runtime overhead may differ.

---

## Training

**Q: Why is training memory larger than inference weight memory?**
A: Training stores gradients, optimizer states, and saved activations.

**Q: What extra tensors does Adam usually keep?**
A: First and second moment estimates.

**Q: Can activation memory dominate training memory?**
A: Yes, especially with long sequences, large batches, or deep models.

---

## Inference

**Q: What is the KV cache?**
A: Stored keys and values from previous tokens for autoregressive attention.

**Q: How does KV cache memory scale with context length?**
A: Linearly.

**Q: Does quantizing weights always quantize the KV cache?**
A: No. KV cache dtype is a separate implementation choice.

---

## Gotchas

**Q: Is parameter memory a complete GPU memory estimate?**
A: No. It is a lower bound.

**Q: How many bytes per parameter is int8?**
A: 1 byte.

**Q: How many bytes per parameter is ideal int4?**
A: 0.5 bytes, before metadata and packing overhead.
