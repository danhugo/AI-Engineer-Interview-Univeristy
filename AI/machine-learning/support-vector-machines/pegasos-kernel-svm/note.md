# Pegasos Kernel SVM - Interview Knowledge Sheet

## Intuition

Pegasos is a stochastic subgradient method for the SVM objective.

For the linear case, it alternates:

```text
sample an example -> take a hinge-loss subgradient step -> project weights back into a norm ball
```

The projection keeps regularization under control.

---

## 1. Linear Update

With regularization $\lambda$ and step size:

$$
\eta_t = \frac{1}{\lambda t}
$$

If $y_i w^Tx_i < 1$:

$$
w \leftarrow (1 - \eta_t\lambda)w + \eta_t y_i x_i
$$

Otherwise:

$$
w \leftarrow (1 - \eta_t\lambda)w
$$

---

## 2. Kernel Version

The kernelized version represents `w` through coefficients on training examples and uses kernel evaluations instead of explicit dot products.

In interviews, the core idea is still stochastic hinge-loss optimization plus projection.
## References

- scikit-learn Support Vector Machines user guide: https://scikit-learn.org/stable/modules/svm.html
- Pegasos paper: https://collaborate.princeton.edu/en/publications/pegasos-primal-estimated-sub-gradient-solver-for-svm-2/
