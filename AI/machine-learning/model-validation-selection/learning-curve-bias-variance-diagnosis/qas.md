# Learning Curve Generator for Bias-Variance Diagnosis - Q&A

---

## Core Idea

**Q: What does a learning curve vary?**
A: The number of training samples.

**Q: What indicates high bias?**
A: Both train and validation scores are low and close.

**Q: What indicates high variance?**
A: Train score is high and validation score is much lower.

## Usage

**Q: Why average across folds?**
A: To reduce split noise.

**Q: What must stay inside folds?**
A: Preprocessing and fitting.
