# Stratified Train-Test Split - Q&A

---

## Core Idea

**Q: What does stratification preserve?**
A: Class proportions across train and test.

**Q: Why use it for imbalanced data?**
A: To avoid validation sets missing rare classes.

**Q: Does it solve grouped leakage?**
A: No. Use group-aware splits for grouped data.
