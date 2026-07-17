# Layer Normalization for Sequence Data - Q&A

---

## Intuition

**Q: What does layer normalization do in a transformer?**
A: It normalizes each token's hidden vector, then applies learned scale and shift.

**Q: For a tensor shaped `(batch, sequence, hidden)`, which axis is commonly normalized?**
A: The hidden axis.

**Q: Does one token use other tokens to compute its layer norm statistics?**
A: No. Each token vector is normalized independently.

---

## Formula

**Q: What mean is used for one hidden vector?**
A: The average of that vector's hidden features.

**Q: What is the normalized value formula?**
A: `(x - mean) / sqrt(variance + eps)`.

**Q: Why are `gamma` and `beta` included?**
A: They let the model learn useful scale and shift after normalization.

---

## Sequence Details

**Q: If `X` has shape `(B, T, H)`, what shape do the means have before broadcasting?**
A: `(B, T, 1)`.

**Q: What shape do `gamma` and `beta` usually have?**
A: `(H,)`.

**Q: Does changing batch composition change a token's layer norm result?**
A: No, because statistics are computed within that token vector.

---

## Comparisons

**Q: How is layer norm different from batch norm?**
A: Batch norm uses batch statistics; layer norm uses per-example feature statistics.

**Q: Is layer norm behavior different in train and eval mode?**
A: No. It uses current input statistics in both modes.

**Q: Why is that useful for autoregressive inference?**
A: Inference can run with batch size one or changing sequence lengths without relying on running batch statistics.

---

## Gotchas

**Q: Where does epsilon go?**
A: Inside the square root with variance.

**Q: What is a common axis bug?**
A: Accidentally normalizing across batch or time instead of hidden features.

**Q: Does layer norm remove the need for residual connections?**
A: No. They solve different parts of transformer optimization.
