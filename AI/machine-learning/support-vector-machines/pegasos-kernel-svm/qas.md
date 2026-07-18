# Pegasos Kernel SVM - Q&A

---

## Core Idea

**Q: What kind of optimizer is Pegasos?**
A: A stochastic subgradient solver for SVMs.

**Q: What is the step size shape?**
A: `1 / (lambda * t)`.

**Q: Why project?**
A: To enforce the regularization norm constraint.
