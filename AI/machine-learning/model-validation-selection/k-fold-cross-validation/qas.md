# K-Fold Cross-Validation - Q&A

---

## Core Idea

**Q: Why use K-fold CV?**
A: To estimate generalization across multiple train/validation splits.

**Q: What is averaged?**
A: The validation score from each fold.

**Q: What does high fold variance mean?**
A: The result is sensitive to which examples are held out.

## Leakage

**Q: Where should preprocessing be fit?**
A: Inside each training fold only.

**Q: When is ordinary K-fold wrong?**
A: For time series, grouped data, or strong class imbalance without stratification.
