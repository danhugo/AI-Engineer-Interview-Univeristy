# Mixed Precision Training - Q&A

---

## Intuition

**Q: What is mixed precision training?**
A: Training that uses lower precision for many computations while keeping sensitive state or operations in higher precision.

**Q: Why use mixed precision?**
A: To reduce memory use and often speed up training on supported hardware.

**Q: Is mixed precision the same as casting everything to `float16`?**
A: No. Important state and numerically sensitive operations often stay in `float32`.

---

## 1. Dtypes

**Q: What is a risk of `float16`?**
A: Small values can underflow to zero and large values can overflow.

**Q: Why can `bfloat16` be easier to train with?**
A: It has a wider exponent range than `float16`.

**Q: What is a tradeoff of `bfloat16`?**
A: It has fewer mantissa bits than `float16`.

---

## 2. Autocast

**Q: What does autocast do?**
A: It chooses lower precision for suitable operations and keeps other operations in safer precision.

**Q: Where is autocast usually used?**
A: Around the forward pass and loss computation.

**Q: Should you manually cast every parameter to half?**
A: Usually no.

---

## 3. Gradient Scaling

**Q: Why scale the loss?**
A: To make tiny `float16` gradients large enough to avoid underflow during backpropagation.

**Q: What must happen before the optimizer step?**
A: The scaled gradients must be unscaled.

**Q: What does dynamic loss scaling do?**
A: It adjusts the scale and can skip updates when gradients overflow.

---

## 4. Master Weights

**Q: Why keep master weights in `float32`?**
A: Small optimizer updates and accumulated optimizer state need more precision.

**Q: What state is especially sensitive?**
A: Optimizer state such as Adam's moving averages.

---

## Interview Gotchas

**Q: What is the key interview one-liner?**
A: AMP trades precision carefully: low precision for speed and memory, high precision for stability.

**Q: What should you monitor after enabling AMP?**
A: NaNs, overflows, loss curves, and final metrics.
