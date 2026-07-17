# KL Divergence — Q&A

---

## Intuition

**Q: What does KL divergence compare?**
A: Two probability distributions.

**Q: Is KL symmetric?**
A: No.

**Q: When is KL zero?**
A: When the two distributions match.

---

## 1. Formula

**Q: Which distribution is usually the target?**
A: `p`.

**Q: Which distribution is the approximation?**
A: `q`.

**Q: What does KL punish?**
A: Using `q` where `p` is the target.

---

## 2. Cross-Entropy Connection

**Q: How are cross-entropy and KL related?**
A: Cross-entropy equals entropy of `p` plus KL from `p` to `q`.

**Q: If `p` is fixed, what does minimizing cross-entropy do?**
A: It also minimizes KL divergence.

---

## 3. PyTorch Pattern

**Q: What input does `KLDivLoss` expect?**
A: Log-probabilities.

**Q: What target does it expect by default?**
A: Probabilities.

**Q: What reduction is often mathematically correct?**
A: `batchmean`.

---

## 4. Interview Gotchas

**Q: When is KL useful?**
A: Soft targets, distillation, and distribution matching.

**Q: What is a common PyTorch mistake?**
A: Passing probabilities as input instead of log-probabilities.
