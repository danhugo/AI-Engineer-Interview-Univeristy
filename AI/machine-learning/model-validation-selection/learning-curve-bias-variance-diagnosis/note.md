# Learning Curve Generator for Bias-Variance Diagnosis - Interview Knowledge Sheet

## Intuition

A learning curve shows training and validation score as training-set size grows.

It helps answer:

```text
Do we need more data, a simpler model, or a more expressive model?
```

---

## 1. Bias-Variance Signals

High bias / underfitting:

- training score is low
- validation score is low
- the gap is small

High variance / overfitting:

- training score is high
- validation score is much lower
- more data may reduce the gap

---

## 2. Procedure

For each training size:

1. train on a subset of that size inside each fold
2. score on that training subset
3. score on the validation fold
4. average across folds

The curve is only meaningful if preprocessing and model fitting happen inside the fold.
## References

- scikit-learn model selection user guide: https://scikit-learn.org/stable/model_selection.html
- scikit-learn learning curve docs: https://scikit-learn.org/stable/modules/learning_curve.html
