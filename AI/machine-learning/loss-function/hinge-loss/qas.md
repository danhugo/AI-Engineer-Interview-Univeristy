# Hinge Loss — Q&A

---

## Intuition

**Q: What kind of loss is hinge loss?**
A: A margin-based classification loss.

**Q: What does it require beyond being correct?**
A: The prediction must be correct by a large enough margin.

**Q: What model is hinge loss associated with?**
A: Support vector machines.

---

## 1. Binary Hinge Loss

**Q: What labels does binary hinge usually use?**
A: `-1` and `+1`.

**Q: What is the binary hinge formula?**
A: `max(0, 1 - y * score)`.

**Q: When is binary hinge loss zero?**
A: When `y * score >= 1`.

---

## 2. Margin Intuition

**Q: What does `y * score` measure?**
A: Whether the score is on the correct side and by how much.

**Q: Can a correct prediction still have hinge loss?**
A: Yes, if the margin is too small.

**Q: What happens after the margin is satisfied?**
A: The loss becomes zero.

---

## 3. Multi-Class Hinge Loss

**Q: What should the true class score beat?**
A: Every wrong class score by at least the margin.

**Q: Does hinge loss need probabilities?**
A: No. It uses raw scores.

---

## 4. When to Use

**Q: When is hinge loss natural?**
A: For SVM-style margin classifiers.

**Q: What is more common for neural classifiers?**
A: Cross-entropy.

---

## 5. Interview Gotchas

**Q: What is a common label mistake?**
A: Using `0/1` labels instead of `-1/+1` for binary hinge.

**Q: How does hinge differ from cross-entropy?**
A: Hinge stops pushing after the margin is satisfied.
