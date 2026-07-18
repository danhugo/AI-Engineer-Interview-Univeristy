# SVM Margin Width - Interview Knowledge Sheet

## Intuition

An SVM wants a decision boundary that separates classes with the widest possible margin.

For a linear boundary:

$$
f(x)=w^Tx+b
$$

The geometric distance from a point to the hyperplane is:

$$
\frac{|w^Tx+b|}{\|w\|}
$$

---

## 1. Margin Width

In the canonical hard-margin setup, support vectors satisfy:

$$
y_i(w^Tx_i+b)=1
$$

The two margin planes are one unit away in functional margin, so the full margin width is:

$$
\frac{2}{\|w\|}
$$

Maximizing margin is equivalent to minimizing $\frac{1}{2}\|w\|^2$.

---

## 2. Soft Margin

Real data may not be perfectly separable. Soft-margin SVM uses `C` to trade off wide margin against margin violations.
## References

- scikit-learn Support Vector Machines user guide: https://scikit-learn.org/stable/modules/svm.html
- Pegasos paper: https://collaborate.princeton.edu/en/publications/pegasos-primal-estimated-sub-gradient-solver-for-svm-2/
