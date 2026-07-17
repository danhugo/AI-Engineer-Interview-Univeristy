# Multi-Head Attention From Scratch - Q&A

---

## Intuition

**Q: What does attention compute?**
A: A weighted sum of value vectors, where weights come from query-key similarity.

**Q: What are queries, keys, and values?**
A: Queries ask what a token needs, keys describe what each token can match, and values carry the information to aggregate.

**Q: Why use multiple heads?**
A: Multiple heads let the model attend to different relationship patterns in parallel.

---

## Formula

**Q: What is scaled dot-product attention?**
A: $\text{softmax}(QK^\top / \sqrt{d_k})V$.

**Q: Why divide by $\sqrt{d_k}$?**
A: To keep dot-product logits from growing too large as dimension increases.

**Q: Which axis does softmax use?**
A: The key axis for each query.

---

## Shapes

**Q: If $d_\text{model}=512$ and $h=8$, what is each head dimension?**
A: $64$.

**Q: What shape do split heads often use?**
A: `(batch, heads, seq_len, head_dim)`.

**Q: What happens after all heads are computed?**
A: They are concatenated and passed through an output projection.

---

## Masks

**Q: What does a causal mask prevent?**
A: A token attending to future tokens.

**Q: Where is a mask applied?**
A: To attention logits before softmax.

**Q: What value is commonly used for disallowed logits?**
A: A very negative value, effectively $-\infty$.

---

## Interview Gotchas

**Q: Do attention weights multiply keys or values?**
A: Values.

**Q: Does multi-head attention increase output dimension by default?**
A: Usually no. Heads split the model dimension and are concatenated back to $d_\text{model}$.
