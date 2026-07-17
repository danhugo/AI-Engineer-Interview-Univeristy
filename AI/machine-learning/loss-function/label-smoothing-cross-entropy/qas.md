# Label Smoothing Cross-Entropy — Q&A

---

## Intuition

**Q: What does label smoothing change?**
A: It changes hard one-hot labels into softer target distributions.

**Q: Why use it?**
A: To reduce overconfidence and regularize classification.

**Q: What is the risk of too much smoothing?**
A: Underfitting.

---

## 1. Smoothed Labels

**Q: What does epsilon control?**
A: How much probability mass is moved away from the true class.

**Q: Do wrong classes stay exactly zero?**
A: No. They get a small target probability.

---

## 2. Loss

**Q: What loss uses the smoothed labels?**
A: Cross-entropy.

**Q: What does smoothing prevent?**
A: Rewarding infinite confidence in the true class.

---

## 3. PyTorch Pattern

**Q: What PyTorch argument enables label smoothing?**
A: `label_smoothing` in `CrossEntropyLoss`.

**Q: Does PyTorch still expect logits?**
A: Yes.

---

## 4. Interview Gotchas

**Q: What is the shortest summary?**
A: Label smoothing is cross-entropy with softened targets.
