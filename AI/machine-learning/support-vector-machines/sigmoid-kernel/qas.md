# Sigmoid Kernel - Q&A

---

## Core Idea

**Q: What is the sigmoid kernel formula?**
A: `tanh(gamma * dot(x,z) + coef0)`.

**Q: Why can it saturate?**
A: Large inputs push tanh near -1 or 1.

**Q: Is it usually the first SVM kernel to try?**
A: No. Linear and RBF are more common defaults.
