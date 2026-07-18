# Bias-Variance Decomposition from Bootstrap - Interview Knowledge Sheet

## Intuition

Prediction error comes from three sources:

```text
systematic wrongness + sensitivity to training data + irreducible noise
```

Bias measures whether the average model prediction is wrong.

Variance measures whether predictions change a lot across different training samples.

Bootstrap resampling approximates those different training samples.

---

## 1. Squared Error View

At one input point $x$:

$$
E[(Y - \hat{f}(x))^2] = Bias^2 + Variance + Noise
$$

Bootstrap estimates the first two terms by fitting many models on resampled datasets and collecting their predictions at the same points.

---

## 2. Estimates

For predictions $\hat{y}_{b,i}$ from bootstrap model `b` on point `i`:

$$
Bias_i^2 = (\bar{\hat{y}}_i - y_i)^2
$$

$$
Variance_i = \frac{1}{B}\sum_b(\hat{y}_{b,i} - \bar{\hat{y}}_i)^2
$$

Average over points for a dataset-level summary.

---

## 3. Interview Framing

High bias means the model class is too simple or underfit. High variance means it is too sensitive to the training sample or overfit.
## References

- scikit-learn model selection user guide: https://scikit-learn.org/stable/model_selection.html
- scikit-learn learning curve docs: https://scikit-learn.org/stable/modules/learning_curve.html
