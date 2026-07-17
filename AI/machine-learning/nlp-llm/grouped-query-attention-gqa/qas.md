# Grouped Query Attention GQA - Q&A

---

## Intuition

**Q: What does GQA reduce?**
A: The number of key/value heads used by attention.

**Q: What does GQA keep?**
A: Multiple query heads.

**Q: Why does this help decoder inference?**
A: It reduces KV cache size and memory bandwidth when attending to previous tokens.

---

## Relationships

**Q: When is GQA equivalent to MHA?**
A: When the number of KV heads equals the number of query heads.

**Q: When is GQA equivalent to MQA?**
A: When there is exactly one KV head.

**Q: What is the group size?**
A: $h_q / h_{kv}$, the number of query heads sharing each KV head.

---

## Shapes

**Q: If there are 16 query heads and 4 KV heads, how many query heads share each KV head?**
A: 4.

**Q: What divisibility condition is common?**
A: Query heads should be divisible by KV heads.

**Q: Does GQA change the model output dimension?**
A: No. The attended heads are still combined back to the model dimension.

---

## Inference

**Q: Why does KV cache memory scale with KV heads?**
A: The cache stores keys and values for every layer, token, KV head, and head dimension.

**Q: What is the main quality-speed tradeoff?**
A: Fewer KV heads improve efficiency but can reduce expressiveness.

**Q: What did the GQA paper propose for existing MHA checkpoints?**
A: Convert/pool KV heads and uptrain with a small amount of additional compute.

---

## Interview Gotchas

**Q: Does GQA make attention approximate?**
A: It changes the parameter sharing pattern, but attention is still exact over the available keys and values.

**Q: Does GQA remove causal masking?**
A: No. Decoder models still use causal masks.
