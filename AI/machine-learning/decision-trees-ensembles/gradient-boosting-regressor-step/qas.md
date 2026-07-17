# Gradient Boosting Regressor Step - Q&A

---

## Core Idea

**Q: What does each new tree fit?**
A: Residuals or negative gradients.

**Q: What is the squared-error residual?**
A: `y - prediction`.

**Q: What does learning rate do?**
A: Shrinks each tree contribution.
