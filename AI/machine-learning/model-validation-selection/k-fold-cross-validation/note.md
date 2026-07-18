# K-Fold Cross-Validation - Interview Knowledge Sheet

## Intuition

A single train/test split can be lucky or unlucky.

K-fold cross-validation reduces that dependence by rotating which part of the dataset acts as validation.

```text
split data into k folds -> train on k-1 folds -> validate on the held-out fold -> repeat
```

The final score is usually the mean and standard deviation across folds.

---

## 1. Formula

For fold scores $s_1, \ldots, s_k$:

$$
\bar{s} = \frac{1}{k}\sum_{j=1}^{k}s_j
$$

The standard deviation tells you how sensitive the estimate is to the split.

---

## 2. What It Estimates

Cross-validation estimates how the modeling procedure performs on unseen data.

The procedure includes preprocessing, feature selection, model fitting, and hyperparameter choice when those steps happen inside each fold.

If preprocessing is fit before CV, validation data leaks into training.

---

## 3. Practical Notes

Use ordinary K-fold for regression and balanced classification. Use stratified K-fold when class proportions matter. Use grouped or time-aware splits when rows are not independent.
## References

- scikit-learn model selection user guide: https://scikit-learn.org/stable/model_selection.html
- scikit-learn learning curve docs: https://scikit-learn.org/stable/modules/learning_curve.html
