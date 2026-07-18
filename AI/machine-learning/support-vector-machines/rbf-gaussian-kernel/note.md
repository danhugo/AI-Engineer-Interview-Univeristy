# RBF Gaussian Kernel - Interview Knowledge Sheet

## Intuition

The RBF kernel measures local similarity.

Two points are similar if they are close in Euclidean distance:

$$
K(x,z) = \exp(-\gamma \|x-z\|^2)
$$

The value is near 1 for very close points and near 0 for far points.

---

## 1. Gamma

`gamma` controls how quickly similarity decays with distance.

Large `gamma` means each point has a small area of influence. This can create a wiggly boundary and overfit.

Small `gamma` means each point influences a wider area. This creates a smoother boundary.

---

## 2. Scaling

RBF kernels are distance-based, so feature scaling is critical. A feature with a large numeric range can dominate the distance.
## References

- scikit-learn Support Vector Machines user guide: https://scikit-learn.org/stable/modules/svm.html
- Pegasos paper: https://collaborate.princeton.edu/en/publications/pegasos-primal-estimated-sub-gradient-solver-for-svm-2/
