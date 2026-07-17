# Focal Loss — Q&A

---

## Intuition

**Q: What does focal loss focus on?**
A: Hard examples that the model does not classify well.

**Q: What does it down-weight?**
A: Easy examples with high true-class probability.

**Q: When is focal loss useful?**
A: Imbalanced classification, especially dense detection.

---

## 1. Formula

**Q: What is `p_t`?**
A: The probability assigned to the true class.

**Q: What does `gamma` control?**
A: How strongly easy examples are down-weighted.

**Q: What happens when `gamma = 0`?**
A: Focal loss becomes weighted cross-entropy.

---

## 2. Why It Works

**Q: What happens when `p_t` is near 1?**
A: The focal weight becomes small.

**Q: What happens when `p_t` is small?**
A: The example keeps a large loss weight.

---

## 3. PyTorch Pattern

**Q: What stable base loss should you use for binary focal loss?**
A: `binary_cross_entropy_with_logits` with `reduction="none"`.

**Q: Why use `reduction="none"`?**
A: You need per-sample losses before applying focal weights.

---

## 4. Interview Gotchas

**Q: What is the core focal-loss summary?**
A: Cross-entropy that focuses learning on hard examples.

**Q: What is a common implementation detail?**
A: Use logits for numerical stability.
