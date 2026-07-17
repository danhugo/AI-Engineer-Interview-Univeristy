# Random Forest Feature Importance - Q&A

---

## Core Idea

**Q: What is impurity importance?**
A: Normalized total impurity reduction credited to a feature.

**Q: How is it aggregated in a forest?**
A: Sum or average credits across trees.

**Q: What is a caveat?**
A: High-cardinality features can look too important.

## Alternative

**Q: What is permutation importance?**
A: Shuffle one feature and measure validation score drop.
