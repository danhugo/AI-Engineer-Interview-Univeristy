# Gradient Clipping — Q&A

---

## Core Idea

**Q: What is gradient clipping?**
A: Limiting gradients before the optimizer update.

**Q: Does clipping change the loss function?**
A: No. It changes the gradient used for the update.

**Q: When is clipping useful?**
A: When gradients can explode and cause unstable updates.

---

## Clip by Norm

**Q: What does norm clipping limit?**
A: The total norm of the gradient vector.

**Q: What happens if the norm is below the threshold?**
A: The gradient is left unchanged.

**Q: What happens if the norm is above the threshold?**
A: The whole gradient is scaled down.

**Q: Does norm clipping preserve direction?**
A: Yes, when it rescales the whole vector.

---

## Clip by Value

**Q: What does value clipping limit?**
A: Each gradient element independently.

**Q: Why can value clipping change direction?**
A: Coordinates may be clamped by different amounts.

---

## PyTorch

**Q: When do you call `clip_grad_norm_`?**
A: After `loss.backward()` and before `optimizer.step()`.

**Q: Does `clip_grad_norm_` modify gradients in place?**
A: Yes.

**Q: What does `clip_grad_norm_` return?**
A: The total gradient norm before clipping.

---

## Gotchas

**Q: Is clipping a replacement for learning-rate tuning?**
A: No. It is a stabilizer, not a full fix.

**Q: What might it mean if clipping happens every step?**
A: The learning rate may be too high or the model may be unstable.
