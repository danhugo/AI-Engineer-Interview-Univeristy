# Linear Kernel - Interview Knowledge Sheet

## Intuition

A linear kernel measures similarity with a dot product.

$$
K(x, z) = x^Tz
$$

If two vectors point in a similar direction and have large aligned values, their dot product is large.

---

## 1. Why It Matters

A linear SVM learns a separating hyperplane:

$$
f(x) = w^Tx + b
$$

The linear kernel lets the dual SVM express that same hyperplane using training examples:

$$
f(x) = \sum_{i \in SV}\alpha_i y_i K(x_i, x) + b
$$

For high-dimensional sparse data, such as text features, linear kernels are often strong and efficient.

---

## 2. When To Use

Use a linear kernel when the relationship is close to linear, features are already expressive, or the dataset is large. Nonlinear kernels can be more flexible but usually cost more.
## References

- scikit-learn Support Vector Machines user guide: https://scikit-learn.org/stable/modules/svm.html
- Pegasos paper: https://collaborate.princeton.edu/en/publications/pegasos-primal-estimated-sub-gradient-solver-for-svm-2/
