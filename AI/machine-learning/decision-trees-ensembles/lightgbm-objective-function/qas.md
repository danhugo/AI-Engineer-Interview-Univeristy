# LightGBM Objective Function - Q&A

---

## Core Idea

**Q: What is LightGBM?**
A: A histogram-based gradient boosted tree system.

**Q: What is leaf-wise growth?**
A: Split the leaf with the largest gain.

**Q: Why use histograms?**
A: To aggregate gradient/Hessian sums and speed split search.
