# FlashAttention Concept and Memory Complexity - Q&A

---

## Intuition

**Q: What problem does FlashAttention solve?**
A: Standard attention writes and reads large $n \times n$ intermediate matrices, making long-sequence attention memory- and IO-heavy.

**Q: Is FlashAttention approximate?**
A: No. It computes exact softmax attention.

**Q: What is the main implementation idea?**
A: Tile attention into blocks and use online softmax so the full attention matrix does not need to be stored.

---

## Complexity

**Q: Does FlashAttention make dense attention compute subquadratic?**
A: No. Dense attention still has $O(n^2d)$ compute.

**Q: What memory term does it avoid materializing?**
A: The $n \times n$ score/probability matrix.

**Q: Why can it be faster if FLOPs are similar?**
A: It reduces slow HBM reads and writes, which often dominate runtime.

---

## Online Softmax

**Q: Why does blockwise softmax need a running max?**
A: For numerical stability and to combine blocks under a shared normalization.

**Q: What running values are tracked?**
A: A max, a normalizer, and the weighted value numerator/output.

**Q: What happens when a new block has a larger max?**
A: Previous accumulated values are rescaled to the new max.

---

## Backward

**Q: How does FlashAttention save backward memory?**
A: It recomputes attention blocks instead of storing the full probability matrix.

**Q: What tradeoff does that create?**
A: More recomputation for less memory traffic and activation storage.

---

## Interview Gotchas

**Q: Is FlashAttention the same as sparse attention?**
A: No. Sparse attention changes the attention pattern; FlashAttention changes the kernel for exact dense attention.

**Q: What hardware idea is central to FlashAttention?**
A: Awareness of memory hierarchy, especially HBM versus on-chip SRAM.
