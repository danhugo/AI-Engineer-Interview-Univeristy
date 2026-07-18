# Hinge Loss for SVM - Interview Knowledge Sheet

## Intuition

Hinge loss only penalizes examples that are wrong or too close to the margin.

For labels $y \in \{-1,+1\}$ and score $s=w^Tx+b$:

$$
L = \max(0, 1 - ys)
$$

If `ys >= 1`, the point is correctly classified with enough margin and gets zero loss.

---

## 1. Why It Creates Support Vectors

Only points with nonzero hinge loss or exactly on the margin affect the solution.

These are the support vectors or margin violators.

---

## 2. Objective

A linear soft-margin SVM balances regularization and hinge loss:

$$
\frac{1}{2}\|w\|^2 + C\sum_i \max(0, 1-y_i(w^Tx_i+b))
$$
## References

- scikit-learn Support Vector Machines user guide: https://scikit-learn.org/stable/modules/svm.html
- Pegasos paper: https://collaborate.princeton.edu/en/publications/pegasos-primal-estimated-sub-gradient-solver-for-svm-2/
