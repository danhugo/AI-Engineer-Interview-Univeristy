# Batch Normalization Forward Pass — Q&A

---

## Intuition

**Q: What does batch normalization do?**
A: It normalizes activations, then applies learned scale and shift.

**Q: Why normalize layer activations?**
A: To make optimization easier by keeping activations in a more stable range.

**Q: What are the learned parameters called?**
A: `gamma` and `beta`.

---

## 1. Training Forward Pass

**Q: What statistics are used in training mode?**
A: The current mini-batch mean and variance.

**Q: What is the normalized activation formula?**
A: `(x - batch_mean) / sqrt(batch_var + eps)`.

**Q: What happens after normalization?**
A: Multiply by `gamma` and add `beta`.

---

## 2. Epsilon

**Q: Why add epsilon?**
A: To avoid division by zero and improve numerical stability.

**Q: Where does epsilon go?**
A: Inside the square root with variance.

**Q: Does epsilon learn from data?**
A: No. It is a fixed small constant.

---

## 3. Gamma and Beta

**Q: What does `gamma` control?**
A: Output scale.

**Q: What does `beta` control?**
A: Output shift.

**Q: Why are they important?**
A: They let the layer recover useful scales and offsets after normalization.

---

## 4. Running Statistics

**Q: What statistics are used in eval mode?**
A: Running mean and running variance.

**Q: Why not use the current batch at inference?**
A: Inference should be deterministic and not depend on other examples in the batch.

**Q: What does momentum affect?**
A: How quickly running statistics move toward new batch statistics.

---

## 5. Axis Details

**Q: For `(N, D)` dense inputs, which axis is normalized over?**
A: The batch axis `N`, independently for each feature.

**Q: For `(N, C, H, W)` convolution inputs, what is usually normalized?**
A: Each channel over `N`, `H`, and `W`.

**Q: What is a common implementation bug?**
A: Computing statistics over the wrong axes.

---

## 6. Interview Gotchas

**Q: Is batch norm behavior the same in train and eval mode?**
A: No.

**Q: Can batch norm behave poorly with very small batches?**
A: Yes, because batch statistics are noisy.

**Q: Does batch norm remove the need for learnable layer parameters?**
A: No. It adds its own learnable scale and shift.
