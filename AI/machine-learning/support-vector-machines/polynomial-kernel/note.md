# Polynomial Kernel - Interview Knowledge Sheet

## Intuition

A polynomial kernel lets a linear SVM act like it has polynomial feature interactions, without explicitly building every interaction feature.

$$
K(x,z) = (\gamma x^Tz + r)^d
$$

`d` is the degree and `r` is often called `coef0`.

---

## 1. What It Captures

Degree 2 can capture pairwise interactions. Higher degrees can capture more complex interactions but become easier to overfit.

The kernel trick computes inner products in that expanded feature space directly from the original vectors.

---

## 2. Practical Notes

Polynomial kernels are sensitive to scaling and hyperparameters. Tune `C`, `degree`, `gamma`, and `coef0` with validation data.
## References

- scikit-learn Support Vector Machines user guide: https://scikit-learn.org/stable/modules/svm.html
- Pegasos paper: https://collaborate.princeton.edu/en/publications/pegasos-primal-estimated-sub-gradient-solver-for-svm-2/
