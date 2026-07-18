# Compute-Optimal Model Size IsoFLOP - Q&A

---

## Core Idea

**Q: What does compute optimal mean?**
A: Best model/data allocation for a fixed training compute budget.

**Q: What is the rough dense LM compute formula?**
A: `C ~= 6 * N * D`.

**Q: What happens if N is too large for C?**
A: The model is undertrained on too few tokens.
