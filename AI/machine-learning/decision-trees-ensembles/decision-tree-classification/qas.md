# Decision Tree Classification - Q&A

---

## Core Idea

**Q: What is stored in leaves?**
A: Class counts or probabilities.

**Q: How does prediction work?**
A: Follow tests to a leaf and choose majority class.

**Q: Why is fitting greedy?**
A: Each split is chosen by local impurity reduction.

## Regularization

**Q: Why do deep trees overfit?**
A: They can split until leaves capture noise.

**Q: Name common controls.**
A: `max_depth`, `min_samples_leaf`, `min_samples_split`, pruning.
