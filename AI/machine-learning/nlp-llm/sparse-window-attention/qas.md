# Sparse Window Attention - Q&A

---

## Intuition

**Q: What problem does sparse window attention target?**
A: The quadratic cost of full self-attention on long sequences.

**Q: What is the core idea?**
A: Each query attends only to keys inside a local window instead of all keys.

**Q: Is it still attention?**
A: Yes. It still uses scaled dot-product scores, masking, softmax, and a weighted sum of values.

---

## Complexity

**Q: Why is full attention quadratic in sequence length?**
A: Every one of $n$ queries is compared with every one of $n$ keys.

**Q: What is the rough cost with a fixed window size $w$?**
A: About $O(nwd)$ instead of $O(n^2d)$ if implemented sparsely.

**Q: Does a dense mask automatically save memory?**
A: No. If the full $n \times n$ matrix is still materialized, memory remains quadratic.

---

## Masks

**Q: What does a window mask allow in bidirectional attention?**
A: Keys $j$ where $|i-j| \leq w$.

**Q: What does a causal window mask allow?**
A: Keys from the past local window, usually $i-w \leq j \leq i$.

**Q: How are masked positions handled before softmax?**
A: They are set to a very negative value so their softmax probability is zero.

---

## Tradeoffs

**Q: What is the main downside?**
A: Distant tokens cannot directly attend to each other in one layer.

**Q: How can models recover long-range information?**
A: Deeper layers, global tokens, dilated patterns, occasional full attention, or retrieval.

**Q: When is window attention a good fit?**
A: Long sequences where local context is usually most important.

---

## Interview Gotchas

**Q: Is sparse attention always exact full attention?**
A: No. It is exact attention over a restricted set of edges, not over all tokens.

**Q: Is causal masking the same as window masking?**
A: No. Causal masking prevents future attention; window masking restricts distance. They can be combined.
