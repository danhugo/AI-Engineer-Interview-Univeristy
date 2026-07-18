# Sigmoid Kernel - Interview Knowledge Sheet

## Intuition

The sigmoid kernel has the shape of a neural-network activation applied to a dot product.

$$
K(x,z) = \tanh(\gamma x^Tz + r)
$$

It can model nonlinear similarity, but it is less commonly the default choice than linear or RBF kernels.

---

## 1. Parameters

`gamma` scales the dot product and `coef0` shifts it.

Because `tanh` saturates, very large positive or negative inputs can make many similarities look almost the same.

---

## 2. Practical Notes

The sigmoid kernel is not positive semidefinite for every parameter setting. In interviews, mention it as an available SVM kernel but usually discuss linear and RBF first.
## References

- scikit-learn Support Vector Machines user guide: https://scikit-learn.org/stable/modules/svm.html
- Pegasos paper: https://collaborate.princeton.edu/en/publications/pegasos-primal-estimated-sub-gradient-solver-for-svm-2/
