# RMSNorm - Q&A

---

## Intuition

**Q: What does RMSNorm normalize by?**
A: The root mean square of the hidden vector.

**Q: What key operation from LayerNorm does RMSNorm remove?**
A: Mean subtraction.

**Q: Does RMSNorm force each vector to have zero mean?**
A: No.

---

## Formula

**Q: What is the RMS formula?**
A: `sqrt(mean(x^2) + eps)`.

**Q: What is the RMSNorm output formula?**
A: `gamma * x / rms(x)`.

**Q: Where is epsilon added?**
A: Inside the square root.

---

## Shapes

**Q: For `(B, T, H)` sequence data, which axis is usually normalized?**
A: The hidden axis `H`.

**Q: What shape does `gamma` usually have?**
A: `(H,)`.

**Q: Does one token use other tokens to compute RMS?**
A: No. Each token vector is normalized independently.

---

## Comparisons

**Q: How does RMSNorm differ from LayerNorm?**
A: RMSNorm rescales without recentering; LayerNorm recenters and rescales.

**Q: Does RMSNorm usually include a bias?**
A: Often no, though implementations can vary.

**Q: Why can RMSNorm be faster than LayerNorm?**
A: It computes fewer statistics and skips mean subtraction.

---

## Gotchas

**Q: Is RMS the same as variance?**
A: No. RMS is based on mean squared values, not squared deviations from the mean.

**Q: What happens to an all-zero vector without epsilon?**
A: Division by zero.

**Q: Does RMSNorm change the output shape?**
A: No. It preserves the input shape.
