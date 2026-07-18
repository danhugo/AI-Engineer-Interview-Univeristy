# MinMaxScaler - Q&A

---

## Core Idea

**Q: What range is common?**
A: `[0, 1]`.

**Q: What values are learned during fit?**
A: Feature-wise min and max from training data.

**Q: Is it robust to outliers?**
A: No. Extremes set the scale.
