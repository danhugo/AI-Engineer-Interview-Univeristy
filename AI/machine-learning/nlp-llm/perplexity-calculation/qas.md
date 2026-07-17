# Perplexity Calculation - Q&A

---

## Intuition

**Q: What does perplexity measure?**
A: How surprised a language model is by the observed tokens.

**Q: Is lower perplexity better?**
A: Yes.

**Q: What is a common intuition for perplexity?**
A: The model's average branching factor.

---

## 1. Formula

**Q: How do you compute perplexity from average NLL with natural logs?**
A: `exp(avg_nll)`.

**Q: What is average NLL?**
A: The mean negative log probability assigned to the true tokens.

**Q: What is perplexity in terms of true-token probabilities?**
A: The geometric mean of inverse true-token probabilities.

---

## 2. Cross-Entropy

**Q: How is perplexity related to cross-entropy?**
A: It is exponentiated cross-entropy.

**Q: What if cross-entropy is measured in bits?**
A: Use `2 ** cross_entropy`.

**Q: What log base do deep learning libraries usually use?**
A: Natural logs.

---

## 3. Logits and Masks

**Q: What stable operation should you use with logits?**
A: Log-softmax or log-sum-exp.

**Q: Which tokens should padding contribute to?**
A: None; padding should be masked out.

**Q: In PyTorch language modeling, what label is often ignored?**
A: `-100`.

---

## 4. Comparisons

**Q: Can you compare perplexity across tokenizers freely?**
A: No.

**Q: Why not?**
A: Tokenization changes the number and identity of prediction events.

**Q: Does low perplexity guarantee better instruction following?**
A: No.
