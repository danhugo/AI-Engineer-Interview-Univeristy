# Gradient Boosting Regressor Step - Interview Knowledge Sheet

## Intuition

Gradient boosting builds an additive model one small tree at a time. Each new tree tries to correct the current model's errors.

For squared error, those errors are residuals:

$$
r_i = y_i - \hat{y}_i
$$

The update is:

$$
F_{t+1}(x) = F_t(x) + \eta h_t(x)
$$

---

## 1. Negative Gradient View

For loss $L(y, F(x))$:

$$
r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}
$$

For squared error, this is proportional to `y - prediction`. Bagging trains independent models; boosting trains sequential models that target previous mistakes.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
