# GPT Feed-Forward Block - Q&A

---

## Intuition

**Q: What does the feed-forward block do after attention?**
A: It transforms each token's hidden features independently.

**Q: Does the feed-forward block mix information across tokens?**
A: No. Attention does that.

**Q: Are the feed-forward weights different for each position?**
A: No. The same weights are reused at every position.

---

## Shapes

**Q: If input shape is `(B, T, D)`, what is output shape?**
A: `(B, T, D)`.

**Q: Why must the output hidden size equal `D`?**
A: So it can be added to the residual stream.

**Q: What is a common inner dimension?**
A: About `4D`.

---

## Formula

**Q: What is the basic two-layer feed-forward formula?**
A: `activation(x @ W1 + b1) @ W2 + b2`.

**Q: What activation is common in GPT-style MLPs?**
A: GELU or a GELU-family activation.

**Q: What are the usual shapes of `W1` and `W2`?**
A: `W1` is `(D, M)` and `W2` is `(M, D)`.

---

## Parameters

**Q: How many weight parameters are in the two projections?**
A: `D * M + M * D`.

**Q: How many bias parameters are added if both layers use bias?**
A: `M + D`.

**Q: If `M = 4D`, roughly how many weight parameters does the MLP have?**
A: About `8D^2`.

---

## Gotchas

**Q: Is the MLP equivalent to attention?**
A: No. It is position-wise feature processing.

**Q: Can the MLP be computed for all tokens in parallel?**
A: Yes, because each token uses the same independent transformation.

**Q: What breaks if the second projection returns `M` instead of `D`?**
A: The residual addition cannot be performed without another projection.
