# XGBoost Objective Function - Q&A

---

## Core Idea

**Q: What does the objective combine?**
A: Training loss and tree complexity penalty.

**Q: What are G and H?**
A: Leaf sums of gradients and Hessians.

**Q: What is the optimal leaf weight?**
A: `-G / (H + lambda)`.
