# Gini Impurity - Q&A

---

## Core Idea

**Q: What does Gini impurity measure?**
A: How mixed the classes are inside a node.

**Q: What is Gini for a pure node?**
A: `0`.

**Q: What formula should you remember?**
A: `1 - sum(p_k ** 2)`.

## Splits

**Q: How is Gini used for splits?**
A: Compare parent impurity to weighted child impurity.

**Q: What split is preferred?**
A: The split with the largest impurity reduction.
