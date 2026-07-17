# Gradient Checkpointing - Q&A

---

## Intuition

**Q: What does gradient checkpointing save?**
A: Memory, by saving fewer activations from the forward pass.

**Q: What is the cost?**
A: Extra compute, because parts of the forward pass are recomputed during backward.

**Q: What is another name for it?**
A: Activation checkpointing.

---

## 1. Normal Autograd

**Q: Why does normal backpropagation need memory?**
A: It stores intermediate activations needed to compute gradients.

**Q: Why can this become a bottleneck?**
A: Large models, long sequences, and large batches can create many large activations.

---

## 2. Checkpointing Idea

**Q: What is saved in a checkpointed segment?**
A: The segment inputs and selected outputs, not every intermediate activation.

**Q: What happens during backward?**
A: The segment forward computation is rerun to recreate needed activations.

**Q: Should gradients match ordinary training?**
A: Yes, when the checkpointed function is valid and randomness is handled correctly.

---

## 3. PyTorch

**Q: Which PyTorch utility is commonly used?**
A: `torch.utils.checkpoint.checkpoint`.

**Q: What argument should modern PyTorch code pass explicitly?**
A: `use_reentrant`.

**Q: What is `checkpoint_sequential` for?**
A: Splitting a sequential model into checkpointed segments.

---

## 4. Randomness

**Q: Why does dropout matter?**
A: The forward segment is rerun during backward, so random masks must be consistent for equivalent gradients.

**Q: What does PyTorch do by default?**
A: It preserves RNG state for deterministic equivalence.

---

## Interview Gotchas

**Q: Is this the same as saving weights to disk?**
A: No. It is activation memory management inside a training step.

**Q: When is checkpointing most useful?**
A: When activation memory limits batch size, sequence length, or model size.
