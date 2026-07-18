# Train/Val/Test Split with No Leakage - Q&A

---

## Core Idea

**Q: What is leakage?**
A: Validation/test information influencing training.

**Q: What is the safe order?**
A: Split first, fit preprocessing on train only.

**Q: What is validation for?**
A: Choosing hyperparameters and modeling decisions.

## Test Set

**Q: When use the test set?**
A: Once, after decisions are frozen.

**Q: Why use pipelines?**
A: To keep preprocessing inside each fold or split.
