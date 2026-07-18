# One-Hot Encoding - Q&A

---

## Core Idea

**Q: What does one-hot encoding create?**
A: A binary column per category.

**Q: Why not label-code nominal features for linear models?**
A: Integer codes imply an order that may not exist.

**Q: When are categories learned?**
A: During fit on training data.

## Unknowns

**Q: How can unknown categories be handled?**
A: Raise an error or encode as all zeros, depending on policy.
