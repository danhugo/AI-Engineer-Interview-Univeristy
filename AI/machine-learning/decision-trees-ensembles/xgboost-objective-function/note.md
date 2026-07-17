# XGBoost Objective Function - Interview Knowledge Sheet

## Intuition

XGBoost is gradient boosting with a regularized tree objective. At each round, it asks which tree reduces loss while staying simple.

$$
\mathcal{L}^{(t)} = \sum_i l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
$$

$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
$$

---

## 1. Gradients and Hessians

For sample `i`:

$$
g_i = \partial_{\hat{y}} l(y_i, \hat{y}_i), \quad h_i = \partial^2_{\hat{y}} l(y_i, \hat{y}_i)
$$

For a leaf with sums `G` and `H`:

$$
w^* = -\frac{G}{H + \lambda}
$$

---

## 2. Split Gain

$$
Gain = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma
$$

This is why XGBoost can score candidate splits efficiently.
## References

- XGBoost parameters and tree booster docs: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost paper: https://arxiv.org/abs/1603.02754
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
